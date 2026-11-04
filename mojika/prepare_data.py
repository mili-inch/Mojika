"""Render glyphs and encode descriptions into a complete training dataset."""

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file
import torch

from .conditional.condition_stats import compute_condition_means
from .conditional.dataset import ConditionalKanjiDataset
from .encoders import load_encoders


def read_descriptions(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(
            "Descriptions must be a nonempty object: character -> text array"
        )
    for char, descriptions in data.items():
        if len(char) != 1 or not isinstance(descriptions, list) or not descriptions:
            raise ValueError(
                f"Expected one character and a nonempty text array: {char!r}"
            )
        if any(not isinstance(text, str) or not text.strip() for text in descriptions):
            raise ValueError(f"Descriptions must be nonempty strings: {char!r}")
    return data


def validate_font(path: Path, characters):
    with TTFont(path) as font:
        cmap = font.getBestCmap()
        if cmap is None:
            raise ValueError(f"Font has no Unicode character map: {path}")
        missing = [char for char in characters if ord(char) not in cmap]
    if missing:
        raise ValueError(f"Font has no glyph for: {''.join(missing)}")


def render_glyph(char: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    image = Image.new("L", (256, 256), 255)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    if right <= left or bottom <= top or right - left > 256 or bottom - top > 256:
        raise ValueError(f"Glyph does not fit the 256x256 canvas: {char!r}")
    draw.text(
        ((256 - right - left) / 2, (256 - bottom - top) / 2), char, font=font, fill=0
    )
    return image


@torch.inference_mode()
def prepare_dataset(descriptions, font, encoders, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("images", "sarashina_embeddings", "clip_embeddings"):
        (output / directory).mkdir()
    for char, texts in descriptions.items():
        code = f"U+{ord(char):04X}"
        render_glyph(char, font).save(output / "images" / f"{code}.png")
        for index, text in enumerate(texts):
            prompts = (text, f"以下の表す漢字は何でしょう？ {text}答: ")
            for variant, prompt in enumerate(prompts):
                embedding = encoders.gpt(prompt).squeeze(0).cpu().contiguous()
                save_file(
                    {"embedding": embedding},
                    str(
                        output
                        / "sarashina_embeddings"
                        / f"{code}_{2 * index + variant}.safetensors"
                    ),
                )
            embedding = encoders.clip_text(text).cpu().contiguous()
            save_file(
                {"embedding": embedding},
                str(output / "clip_embeddings" / f"{code}_{index}.safetensors"),
            )
        print(f"Prepared {char} ({code})", flush=True)
    save_file(
        compute_condition_means(output), str(output / "condition_means.safetensors")
    )
    return ConditionalKanjiDataset(output, seed=42).validate_all()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("assets/models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Dataset already exists: {args.output_dir}")
    descriptions = read_descriptions(args.descriptions)
    validate_font(args.font, descriptions)
    font = ImageFont.truetype(str(args.font), size=200)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Dataset encoding requires a CUDA GPU with bfloat16 support")
    torch.set_num_threads(4)
    encoders = load_encoders(
        args.models_dir / "sarashina2.2-0.5b",
        args.models_dir / "clip-japanese-base",
        torch.device("cuda"),
    )
    print(json.dumps(prepare_dataset(descriptions, font, encoders, args.output_dir)))


if __name__ == "__main__":
    main()
