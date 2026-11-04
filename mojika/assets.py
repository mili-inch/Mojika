"""Download the published weights and the encoders used by Mojika."""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


WEIGHTS_REPO = "Tonimono/Mojika"
WEIGHTS_SHA256 = "1c0e184cd9ca98869ee70bb20479fc6f6f7c8d9a665e23910caaa60483d1875b"
ENCODERS = (
    ("sbintuitions/sarashina2.2-0.5b", "5fb086c49f49824cfc93f09cc4ed5cd5917bef3d"),
    ("line-corporation/clip-japanese-base", "a8f27cd782df48058849cfff4f64e18e60f06c17"),
)


def verify_weights(path: Path):
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != WEIGHTS_SHA256:
        raise ValueError(f"Weight checksum mismatch: {path} ({checksum})")


def download_encoders(destination: Path):
    for repo, revision in ENCODERS:
        snapshot_download(
            repo,
            revision=revision,
            local_dir=destination / repo.split("/")[1],
            ignore_patterns=["onnx/*", "*.msgpack", "*.h5"],
        )
    # CLIP's upstream custom code resolves these IDs at main in the HF cache.
    # Prepare them under the same HF_HOME used to run the demo.
    snapshot_download("rinna/japanese-clip-vit-b-16", allow_patterns=["config.json"])
    snapshot_download(
        "rinna/japanese-roberta-base",
        allow_patterns=[
            "config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "spiece.model",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--encoders-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.encoders_only:
        path = hf_hub_download(
            WEIGHTS_REPO, "weights.pt", local_dir=args.output_dir / "weights"
        )
        verify_weights(Path(path))
        print(path)
    download_encoders(args.output_dir / "models")


if __name__ == "__main__":
    main()
