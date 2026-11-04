import argparse
import fcntl
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mojika.conditional.checkpoint import load_models, save_checkpoint
from mojika.conditional.conditioner import Conditioner
from mojika.conditional.condition_sampling import epoch_samples
from mojika.conditional.config import ExperimentConfig
from mojika.conditional.dataset import ConditionalKanjiDataset
from mojika.conditional.flow import interpolate_patches, sample_patch_times
from mojika.conditional.metrics import (
    MetricBuffer,
    image_loss_metrics,
    patch_time_metrics,
)
from mojika.conditional.model import DiffusionTransformer
from mojika.conditional.optimizers import Optimizers
from mojika.conditional.sample import save_samples
from mojika.conditional.schedule import WSD


def full_batches_per_epoch(dataset_size: int, batch_size: int) -> int:
    batches = dataset_size // batch_size
    if batches == 0:
        raise ValueError("The dataset must contain at least one full training batch")
    return batches


def epoch_loader(dataset, training, epoch: int, batch_in_epoch: int):
    batches = full_batches_per_epoch(len(dataset), training.batch_size)
    if not 0 <= batch_in_epoch < batches:
        raise ValueError("Batch cursor is outside the training epoch")
    dataset.epoch = epoch
    indices = epoch_samples(len(dataset), training.batch_size, training.seed, epoch)
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        sampler=indices[batch_in_epoch * training.batch_size :],
        drop_last=True,
        num_workers=training.num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(training.seed + epoch),
    )


@dataclass
class TrainingState:
    config: ExperimentConfig
    model: DiffusionTransformer
    conditioner: Conditioner
    optimizers: Optimizers
    schedule: WSD
    dataset: dict
    global_step: int
    epoch: int
    batch_in_epoch: int
    epoch_loss_sum: torch.Tensor
    forked_from: dict | None

    @property
    def batches_per_epoch(self):
        return full_batches_per_epoch(
            self.dataset["counts"]["images"], self.config.training.batch_size
        )

    def record_batch(self, loss: torch.Tensor, batch_size: int) -> torch.Tensor | None:
        if batch_size != self.config.training.batch_size:
            raise ValueError("Training updates require a full batch")
        self.global_step += 1
        self.batch_in_epoch += 1
        self.epoch_loss_sum.add_(loss.detach().to(torch.float64), alpha=batch_size)
        if self.batch_in_epoch == self.batches_per_epoch:
            mean = self.epoch_loss_sum / (self.batches_per_epoch * batch_size)
            self.epoch += 1
            self.batch_in_epoch = 0
            self.epoch_loss_sum.zero_()
            return mean
        return None

    def checkpoint(self, rng_state, cuda_rng_state):
        return {
            "config": self.config.to_dict(),
            "dataset": self.dataset,
            "model_state_dict": self.model.state_dict(),
            "conditioner_state_dict": self.conditioner.state_dict(),
            "optimizer_state_dict": self.optimizers.state_dict(),
            "scheduler_state_dict": self.schedule.to_dict(),
            "progress": {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "epoch_loss_sum": self.epoch_loss_sum.item(),
            },
            "forked_from": self.forked_from,
            "rng_state": rng_state,
            "cuda_rng_state": cuda_rng_state,
        }


def restore_state(checkpoint: dict, device: torch.device) -> TrainingState:
    required = {
        "config",
        "dataset",
        "model_state_dict",
        "conditioner_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "progress",
        "forked_from",
        "rng_state",
        "cuda_rng_state",
    }
    if set(checkpoint) != required:
        raise ValueError("Training checkpoint has missing or unknown fields")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    progress = checkpoint["progress"]
    if set(progress) != {"global_step", "epoch", "batch_in_epoch", "epoch_loss_sum"}:
        raise ValueError("Checkpoint progress has missing or unknown fields")
    for key in ("global_step", "epoch", "batch_in_epoch"):
        if type(progress[key]) is not int or progress[key] < 0:
            raise ValueError(f"Checkpoint {key} must be a nonnegative integer")
    batches = full_batches_per_epoch(
        checkpoint["dataset"]["counts"]["images"], config.training.batch_size
    )
    if progress["batch_in_epoch"] >= batches:
        raise ValueError("Checkpoint batch cursor is outside the training epoch")
    if not math.isfinite(progress["epoch_loss_sum"]) or progress["epoch_loss_sum"] < 0:
        raise ValueError("Checkpoint epoch loss must be finite and nonnegative")
    if progress["batch_in_epoch"] == 0 and progress["epoch_loss_sum"] != 0:
        raise ValueError("Checkpoint epoch loss must be zero at the epoch start")
    model, conditioner = load_models(checkpoint)
    model.to(device)
    conditioner.to(device)
    optimizers = Optimizers(
        model, conditioner, config.optimizer, compile=config.training.compile
    )
    optimizers.load_state_dict(checkpoint["optimizer_state_dict"])
    schedule = WSD.from_dict(checkpoint["scheduler_state_dict"])
    if (
        schedule.base_lrs != optimizers.base_lrs
        or schedule.warmup_steps != config.warmup_steps
    ):
        raise ValueError(
            "Scheduler state disagrees with optimizer or warmup configuration"
        )
    optimizers.validate_schedule(schedule, progress["global_step"])
    return TrainingState(
        config,
        model,
        conditioner,
        optimizers,
        schedule,
        checkpoint["dataset"],
        progress["global_step"],
        progress["epoch"],
        progress["batch_in_epoch"],
        torch.tensor(progress["epoch_loss_sum"], dtype=torch.float64, device=device),
        checkpoint["forked_from"],
    )


def fork_decay(
    state: TrainingState, source: Path, steps: int, curve: str, min_lr_ratio: float
):
    state.schedule = state.schedule.fork_decay(
        state.global_step, steps, curve, min_lr_ratio
    )
    state.forked_from = {
        "checkpoint": str(source.resolve()),
        "global_step": state.global_step,
    }


def clip_and_check_gradients(model, conditioner, max_norm, loss, *, check_loss):
    """Keep separate clipping bounds and check both norms with one host sync."""
    norms = tuple(
        torch.nn.utils.clip_grad_norm_(
            module.parameters(), max_norm, error_if_nonfinite=False, foreach=True
        )
        for module in (model, conditioner)
    )
    checked = (*norms, loss.detach()) if check_loss else norms
    if not torch.isfinite(torch.stack(checked)).all():
        raise FloatingPointError("Non-finite gradient norm or checked loss")
    return norms


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="256px conditional training with Warmup-Stable-Decay"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    stable = subcommands.add_parser(
        "stable",
        help="Train a new model through Warmup and then indefinitely at Stable LR",
    )
    stable.add_argument("--config", type=Path, required=True)
    stable.add_argument("--dataset-root", type=Path, required=True)
    stable.add_argument("--output-dir", type=Path, required=True)
    resume = subcommands.add_parser(
        "resume",
        help="Resume latest.pt with its optimizer, scheduler, cursor and RNG state",
    )
    resume.add_argument("--run-dir", type=Path, required=True)
    decay = subcommands.add_parser(
        "decay", help="Fork a completed-warmup Stable checkpoint into a new run"
    )
    decay.add_argument("--checkpoint", type=Path, required=True)
    decay.add_argument("--output-dir", type=Path, required=True)
    decay.add_argument("--decay-steps", type=int, required=True)
    decay.add_argument("--curve", choices=("linear", "cosine", "sqrt"), required=True)
    decay.add_argument("--min-lr-ratio", type=float, required=True)
    for command in (stable, resume, decay):
        command.add_argument(
            "--max-steps",
            type=int,
            help="Stop at this absolute optimizer step; does not alter the LR schedule",
        )
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max_steps must be positive")
    return args


def run(args, run_dir: Path):
    device = torch.device("cuda")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires CUDA and bfloat16 support")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if args.command == "stable":
        config = ExperimentConfig.read(args.config)
        torch.manual_seed(config.training.seed)
        dataset = ConditionalKanjiDataset(args.dataset_root, seed=config.training.seed)
        full_batches_per_epoch(len(dataset), config.training.batch_size)
        print("Validating all images and condition embeddings...", flush=True)
        counts = dataset.validate_all()
        print(json.dumps({"event": "dataset_validated", **counts}), flush=True)
        model = DiffusionTransformer(**asdict(config.model)).to(device)
        conditioner = Conditioner(
            dataset.gpt_mean,
            dataset.clip_mean,
            embed_dim=config.model.embed_dim,
            patch_grid=config.model.img_size // config.model.patch_size,
            depth=config.model.depth,
        ).to(device)
        optimizers = Optimizers(
            model, conditioner, config.optimizer, compile=config.training.compile
        )
        schedule = WSD(config.warmup_steps, optimizers.base_lrs, None)
        optimizers.apply_schedule(schedule, 0)
        state = TrainingState(
            config,
            model,
            conditioner,
            optimizers,
            schedule,
            {"root": str(dataset.root), "counts": counts},
            0,
            0,
            0,
            torch.zeros((), dtype=torch.float64, device=device),
            None,
        )
    else:
        source = run_dir / "latest.pt" if args.command == "resume" else args.checkpoint
        checkpoint = torch.load(
            source, weights_only=True, map_location="cpu", mmap=True
        )
        state = restore_state(checkpoint, device)
        config = state.config
        dataset = ConditionalKanjiDataset(
            Path(state.dataset["root"]), seed=config.training.seed
        )
        if dataset.counts() != state.dataset["counts"]:
            raise ValueError("Dataset counts differ from the saved training run")
        if args.command == "decay":
            fork_decay(state, source, args.decay_steps, args.curve, args.min_lr_ratio)
        torch.set_rng_state(checkpoint["rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        del checkpoint

    target_step = args.max_steps
    if state.schedule.end_step is not None:
        if target_step is None:
            target_step = state.schedule.end_step
        elif target_step > state.schedule.end_step:
            raise ValueError(
                "max_steps cannot extend a decay branch past its scheduled endpoint"
            )
    if target_step is not None and target_step <= state.global_step:
        raise ValueError(
            "The checkpoint has already reached the requested stopping step"
        )

    training = config.training
    model, conditioner, optimizers = state.model, state.conditioner, state.optimizers
    model.train()
    conditioner.train()
    train_model = torch.compile(model) if training.compile else model
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum}: checkpointing after the current batch", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def persist(archive: bool, samples: bool):
        payload = state.checkpoint(torch.get_rng_state(), torch.cuda.get_rng_state())
        save_checkpoint(run_dir / "latest.pt", payload)
        if archive:
            os.link(run_dir / "latest.pt", run_dir / f"step_{state.global_step:09d}.pt")
        print(
            json.dumps(
                {
                    "event": "checkpoint",
                    "step": state.global_step,
                    "phase": state.schedule.phase(state.global_step),
                }
            ),
            flush=True,
        )
        if samples:
            model.eval()
            conditioner.eval()
            save_samples(
                model,
                conditioner,
                dataset,
                run_dir / f"samples_{state.global_step:09d}.png",
                training.sample_steps,
                training.seed,
            )
            model.train()
            conditioner.train()

    run_metadata = {
        "config": config.to_dict(),
        "dataset": state.dataset,
        "schedule": state.schedule.to_dict(),
        "optimizer_layout": optimizers.layout,
        "forked_from": state.forked_from,
        "pid": os.getpid(),
        "command": args.command,
        "stop_at_step": target_step,
        "batching": {
            "drop_last": True,
            "batches_per_epoch": state.batches_per_epoch,
            "images_per_epoch": state.batches_per_epoch * training.batch_size,
            "discarded_images_per_epoch": len(dataset) % training.batch_size,
            "condition_examples_per_mode_per_batch": training.batch_size // 8,
            "condition_cycle_epochs": 8,
            "embedding_selection": "shuffled_cycle_per_character_and_mode",
            "remainder_selection": "rotating_omission_per_condition_cycle",
        },
        "time_sampling": {
            "distribution": asdict(training.time_sampling),
            "random_region_offset": any(
                config.model.patch_size < size < config.model.img_size
                for size in training.time_patch_sizes
            ),
            "region_offset_unit_pixels": config.model.patch_size,
            "region_boundary": "crop",
            "full_image_region_shared_time": True,
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "started",
                "pid": os.getpid(),
                "command": args.command,
                "step": state.global_step,
                "epoch": state.epoch + 1,
                "batch_in_epoch": state.batch_in_epoch,
                "phase": state.schedule.phase(state.global_step),
                "stop_at_step": target_step,
                "gpu": torch.cuda.get_device_name(),
            }
        ),
        flush=True,
    )
    last_saved_step = state.global_step if args.command == "resume" else None
    if args.command == "decay":
        persist(archive=True, samples=False)
        last_saved_step = state.global_step

    start = time.monotonic()
    start_step = state.global_step
    combinations = {f"{i:03b}": 0 for i in range(8)}
    with (run_dir / "metrics.jsonl").open("a", buffering=1) as metrics:
        pending = MetricBuffer()

        def flush_metrics():
            for record in pending.drain():
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                metrics.write(line + "\n")

        while not stop_requested and (
            target_step is None or state.global_step < target_step
        ):
            epoch = state.epoch
            loader = epoch_loader(dataset, training, epoch, state.batch_in_epoch)
            for batch in loader:
                modes = batch["condition_mode"]
                counts = torch.bincount(modes, minlength=8).tolist()
                if counts != [training.batch_size // 8] * 8:
                    raise RuntimeError(
                        "Training batch does not balance condition modes"
                    )
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                    if key != "unicode"
                }
                x1 = batch["image"]
                if x1.size(0) != training.batch_size:
                    raise RuntimeError(
                        "DataLoader produced an incomplete training batch"
                    )
                x0 = torch.randn_like(x1)
                t, time_patch_sizes, time_patch_offsets = sample_patch_times(
                    x1.size(0),
                    config.model.img_size,
                    config.model.patch_size,
                    training.time_patch_sizes,
                    device,
                    time_sampling=training.time_sampling,
                )
                xt = interpolate_patches(x1, x0, t, config.model.patch_size)
                conditions = {
                    key: batch[key] for key in ("gpt", "clip", "mask", "condition_mode")
                }
                optimizers.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    patch_emb, ext_emb = conditioner(xt, **conditions)
                    pred = train_model(xt, t, patch_emb, ext_emb)
                    loss = F.mse_loss(pred.float(), x1 - x0)
                loss.backward()
                model_grad, cond_grad = clip_and_check_gradients(
                    model,
                    conditioner,
                    training.grad_clip_norm,
                    loss,
                    check_loss=(state.global_step + 1) % training.check_loss_every_steps
                    == 0,
                )
                optimizers.apply_schedule(state.schedule, state.global_step + 1)
                optimizers.step()
                batch_in_epoch = state.batch_in_epoch + 1
                epoch_loss = state.record_batch(loss, x1.size(0))
                for mode, count in enumerate(counts):
                    combinations[f"{mode:03b}"] += count
                pending.append(
                    {
                        "event": "step",
                        "epoch": epoch + 1,
                        "batch_in_epoch": batch_in_epoch,
                        "batch_size": x1.size(0),
                        "step": state.global_step,
                        "phase": state.schedule.phase(state.global_step),
                        "loss": loss,
                        "learning_rates": optimizers.learning_rate_metrics(),
                        "optimizer_hyperparameters": optimizers.hyperparameter_metrics(),
                        "model_grad_norm": model_grad,
                        "conditioner_grad_norm": cond_grad,
                        "condition_example_counts": combinations,
                        **image_loss_metrics(pred, x1 - x0, batch["condition_mode"]),
                        "patch_times": patch_time_metrics(
                            t, time_patch_sizes, time_patch_offsets
                        ),
                        "seconds_per_step": (time.monotonic() - start)
                        / (state.global_step - start_step),
                        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                )
                if epoch_loss is not None:
                    pending.append(
                        {
                            "event": "epoch",
                            "epoch": epoch + 1,
                            "step": state.global_step,
                            "loss": epoch_loss,
                            "images": state.batches_per_epoch * training.batch_size,
                            "batches": state.batches_per_epoch,
                        }
                    )
                finished = target_step is not None and state.global_step == target_step
                if (
                    state.global_step % training.log_every_steps == 0
                    or finished
                    or stop_requested
                ):
                    flush_metrics()
                if (
                    state.global_step % training.save_every_steps == 0
                    or finished
                    or stop_requested
                ):
                    archive = (
                        state.global_step % training.keep_every_steps == 0
                        or finished
                        or stop_requested
                    )
                    samples = (
                        state.global_step % training.sample_every_steps == 0
                        or state.global_step == training.save_every_steps
                        or finished
                        or stop_requested
                    )
                    persist(archive=archive, samples=samples)
                    last_saved_step = state.global_step
                if finished or stop_requested:
                    break
        flush_metrics()
        if last_saved_step != state.global_step:
            persist(archive=True, samples=True)
    print(
        json.dumps(
            {
                "event": "finished",
                "step": state.global_step,
                "phase": state.schedule.phase(state.global_step),
                "checkpoint": str(run_dir / "latest.pt"),
            }
        ),
        flush=True,
    )


def main(args):
    if args.command == "resume":
        run_dir = args.run_dir.resolve(strict=True)
        if not (run_dir / "latest.pt").is_file():
            raise FileNotFoundError(f"No checkpoint in {run_dir}")
    else:
        run_dir = args.output_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another process owns this training run: {run_dir}"
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        run(args, run_dir)


if __name__ == "__main__":
    main(parse_args())
