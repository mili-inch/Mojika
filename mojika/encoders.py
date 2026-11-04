"""Text and image encoders used to construct the training dataset."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer


@dataclass
class Encoders:
    gpt_model: torch.nn.Module
    gpt_tokenizer: LlamaTokenizer
    clip_model: torch.nn.Module
    clip_tokenizer: object
    clip_processor: object
    device: torch.device

    @torch.inference_mode()
    def gpt(self, text: str) -> torch.Tensor:
        tokens = self.gpt_tokenizer.encode(text, return_tensors="pt")
        if tokens.shape[1] > 1024:
            raise ValueError("GPT入力は1,024トークン以内にしてください。")
        output = self.gpt_model(input_ids=tokens.to(self.device), use_cache=False)
        return output.last_hidden_state[:, -1, :].float()

    @torch.inference_mode()
    def clip_text(self, text: str) -> torch.Tensor:
        inputs = self.clip_tokenizer(text).to(self.device)
        return self.clip_model.get_text_features(**inputs).float()

    @torch.inference_mode()
    def clip_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.clip_processor(images=image.convert("RGB"), return_tensors="pt")
        return self.clip_model.get_image_features(**inputs.to(self.device)).float()


def load_encoders(gpt_path: Path, clip_path: Path, device: torch.device) -> Encoders:
    gpt = (
        LlamaForCausalLM.from_pretrained(
            str(gpt_path.resolve(strict=True)),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    gpt_tokenizer = LlamaTokenizer.from_pretrained(str(gpt_path), local_files_only=True)
    clip = (
        AutoModel.from_pretrained(
            str(clip_path.resolve(strict=True)),
            trust_remote_code=True,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    clip_tokenizer = AutoTokenizer.from_pretrained(
        str(clip_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    clip_processor = AutoImageProcessor.from_pretrained(
        str(clip_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    return Encoders(
        gpt.model, gpt_tokenizer, clip, clip_tokenizer, clip_processor, device
    )
