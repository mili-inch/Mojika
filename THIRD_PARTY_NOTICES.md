# Third-party notices

Mojika's code and published Mojika weights use the [MIT License](LICENSE).
External encoders, fonts and datasets retain their own licenses.

- `mojika/conditional/muon.py` adapts [nanochat](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/optim.py). Copyright (c) 2025 Andrej Karpathy, MIT; the complete notice is in the file.
- `web/src/assets/noto-sans-jp.woff2`: Japanese font subset, copyright 2014–2021 Adobe, Reserved Font Name “Source”, [SIL OFL 1.1](web/src/assets/OFL.txt).
- [Sarashina2.2-0.5B](https://huggingface.co/sbintuitions/sarashina2.2-0.5b): SB Intuitions, MIT; revision `5fb086c49f49824cfc93f09cc4ed5cd5917bef3d`.
- [CLIP Japanese Base](https://huggingface.co/line-corporation/clip-japanese-base): LY Corporation, Apache 2.0; revision `a8f27cd782df48058849cfff4f64e18e60f06c17`. Its custom code also loads configuration/tokenizer files from [rinna/japanese-clip-vit-b-16](https://huggingface.co/rinna/japanese-clip-vit-b-16) and [rinna/japanese-roberta-base](https://huggingface.co/rinna/japanese-roberta-base).

Python and JavaScript dependencies are recorded in `uv.lock` and `web/package-lock.json` and retain their upstream licenses.
