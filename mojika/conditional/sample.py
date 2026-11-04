import json
from pathlib import Path

import torch
from torch.utils.data import default_collate
from torchvision.utils import save_image

from .flow import euler_sample


@torch.inference_mode()
def save_samples(model, conditioner, dataset, output: Path, steps: int, seed: int):
    device = next(model.parameters()).device
    indices = [i * (len(dataset) - 1) // 3 for i in range(4)]
    batch = default_collate([dataset.read_sample(index, 0, 0, 7) for index in indices])
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key not in ("unicode", "condition_mode")
    }
    rows = [inputs["image"].add(1).div(2)]
    labels = ["reference"]
    if conditioner is not None:
        rows.append(
            torch.nn.functional.interpolate(
                inputs["mask"].unsqueeze(1).add(1).div(2),
                size=(256, 256),
                mode="nearest",
            )
        )
        labels.append("partial_mask")
    modes = (
        ["unconditional"]
        if conditioner is None
        else ["unconditional", "gpt", "clip", "mask", "all"]
    )
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        noise = torch.randn(4, 1, 256, 256, device=device)
        for mode in modes:
            gpt = inputs["gpt"] if mode in ("gpt", "all") else None
            clip = inputs["clip"] if mode in ("clip", "all") else None
            mask = inputs["mask"] if mode in ("mask", "all") else None
            with torch.autocast(device.type, dtype=torch.bfloat16):
                if conditioner is None:
                    patch_emb, ext_emb = None, None
                else:
                    patch_emb, ext_emb = conditioner(noise, mask, gpt, clip)
                x = euler_sample(model, noise, patch_emb, ext_emb, steps)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite {mode} sample")
            rows.append(x.add(1).div(2).clamp(0, 1))
    save_image(torch.cat(rows), output, nrow=4)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "columns": batch["unicode"],
                "rows": [*labels, *modes],
                "steps": steps,
                "seed": seed,
            },
            indent=2,
        )
        + "\n"
    )
