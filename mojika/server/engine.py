"""GPU-owned model execution. Called exclusively from one worker thread."""

import hashlib
import time

import torch

from mojika.conditional.checkpoint import load_models
from mojika.conditional.stream import AB2Stream
from mojika.encoders import load_encoders
from .interpolation import Interpolation
from .schema import ModelInfo
from .transfer import TransferBuffer


class Engine:
    @torch.inference_mode()
    def __init__(self, config, progress):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDAとbfloat16に対応するGPUが必要です。")
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        self.config = config
        self.device = torch.device("cuda")
        payload = torch.load(
            config.checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        self.model, self.conditioner = load_models(payload)
        if self.model.img_size != 256 or self.model.in_chans != 1:
            raise ValueError("256×256のグレースケールモデルが必要です。")
        self.model.to(self.device).eval()
        self.conditioner.to(self.device).eval()
        del payload
        self.encoders = load_encoders(config.gpt_model, config.clip_model, self.device)
        self.interpolation = Interpolation(
            {
                "gpt": self.conditioner.gpt_embedder.mean.new_zeros(1, 1280),
                "clip": self.conditioner.clip_embedder.mean.new_zeros(1, 512),
                "hint": self.conditioner.mask_uncond,
            },
            self.device,
            config.seed,
            time.monotonic(),
        )
        self.predict = (
            torch.compile(self.model, dynamic=False) if config.compile else self.model
        )
        self.restart_stream(0)
        self.copy_stream = torch.cuda.Stream()
        self.buffers = [TransferBuffer(), TransferBuffer()]
        progress("warming")
        for _ in range(config.steps + 1):
            if self.step(0, self.buffers[0]):
                self.buffers[0].encode()
        with config.checkpoint.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        self.info = ModelInfo(
            checkpoint=config.checkpoint.name,
            sha256=checksum,
            device=torch.cuda.get_device_name(),
            image_size=256,
            steps=config.steps,
        )

    def embeddings(self):
        values = self.interpolation.values
        return self.conditioner.project_centered(
            values["hint"], values["gpt"], values["clip"]
        )

    def restart_stream(self, revision):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch, ext = self.embeddings()
        self.stream = AB2Stream(
            self.predict,
            self.interpolation.values["noise"],
            patch,
            ext,
            self.config.steps,
        )
        self.tags = [(revision, time.time() * 1000)] * self.config.steps

    @torch.inference_mode()
    def apply(self, kind, value):
        if kind == "gpt":
            tensor = self.conditioner.gpt_embedder.prepare(self.encoders.gpt(value))
            self.interpolation.update("gpt", tensor, time.monotonic())
        elif kind == "clip_text":
            tensor = self.conditioner.clip_embedder.prepare(
                self.encoders.clip_text(value)
            )
            self.interpolation.update("clip", tensor, time.monotonic())
        elif kind == "clip_image":
            tensor = self.conditioner.clip_embedder.prepare(
                self.encoders.clip_image(value.image)
            )
            self.interpolation.update("clip", tensor, time.monotonic())
        elif kind == "hint":
            tensor = (
                torch.tensor(value, device=self.device, dtype=torch.float32).reshape(
                    1, 16, 16
                )
                / 127.5
                - 1
            )
            self.interpolation.update("hint", tensor, time.monotonic())
        else:
            self.interpolation.update(kind, value, time.monotonic())

    @torch.inference_mode()
    def step(self, revision, buffer):
        self.interpolation.advance(time.monotonic())
        head = (self.config.steps - 1 - self.stream.iteration) % self.config.steps
        tag = self.tags[head]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            patch, ext = self.embeddings()
            image = self.stream.step(self.interpolation.values["noise"], patch, ext)
        self.tags[head] = (revision, time.time() * 1000)
        if image is None:
            return False
        buffer.enqueue(image, self.copy_stream, *tag)
        return True
