"""Exercise training, resume, decay and export on CUDA with a small model."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import torch

from mojika.conditional.checkpoint import load_models
from mojika.conditional.flow import euler_sample
from tests.fixtures import small_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = small_config("muon_adamw")
    config = replace(
        config, training=replace(config.training, batch_size=8, check_loss_every_steps=1)
    )
    data = config.to_dict()
    lines = [f"warmup_steps = {config.warmup_steps}"]
    for section in ("model", "optimizer", "training"):
        lines += ["", f"[{section}]"]
        lines += [
            f"{key} = {json.dumps(value)}"
            for key, value in data[section].items()
            if key != "time_sampling"
        ]
    lines += ["", "[training.time_sampling]"]
    lines += [
        f"{key} = {json.dumps(value)}"
        for key, value in asdict(config.training.time_sampling).items()
    ]
    config_path = output / "train.toml"
    config_path.write_text("\n".join(lines) + "\n")
    commands = [
        [
            "train.py",
            "stable",
            "--config",
            str(config_path),
            "--dataset-root",
            str(args.dataset_root.resolve(strict=True)),
            "--output-dir",
            str(output / "stable"),
            "--max-steps",
            "3",
        ],
        ["train.py", "resume", "--run-dir", str(output / "stable"), "--max-steps", "4"],
        [
            "train.py",
            "decay",
            "--checkpoint",
            str(output / "stable/latest.pt"),
            "--output-dir",
            str(output / "decay"),
            "--decay-steps",
            "2",
            "--curve",
            "cosine",
            "--min-lr-ratio",
            "0.01",
        ],
        [
            "-m",
            "mojika.conditional.export",
            "--checkpoint",
            str(output / "decay/latest.pt"),
            "--output",
            str(output / "weights.pt"),
        ],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=root, check=True)
    checkpoint = torch.load(
        output / "decay/latest.pt", weights_only=True, map_location="cpu"
    )
    assert checkpoint["progress"]["global_step"] == 6
    payload = torch.load(output / "weights.pt", weights_only=True, map_location="cpu")
    assert set(payload) == {"config", "model_state_dict", "conditioner_state_dict"}
    model, conditioner = load_models(payload)
    model.cuda().eval()
    conditioner.cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        noise = torch.randn(1, 1, 256, 256, device="cuda")
        patch, extra = conditioner(noise)
        image = euler_sample(model, noise, patch, extra, 2)
    assert image.shape == (1, 1, 256, 256) and torch.isfinite(image).all()
    report = {
        "stable_steps": 3,
        "resumed_steps": 4,
        "decay_steps": 6,
        "export_and_inference": "passed",
        "device": torch.cuda.get_device_name(),
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
