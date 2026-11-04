"""Double-buffered CUDA → pinned host transfer; CPU access follows CUDA events."""

import time

import torch

from .frames import EncodedImage, encode_png


def display_pixels(image):
    # Invert after rounding to preserve the exact display polarity at half ties.
    return 255 - ((image.float() + 1) * 127.5).clamp(0, 255).round().to(torch.uint8)


class TransferBuffer:
    def __init__(self):
        self.pixels = torch.empty((256, 256), dtype=torch.uint8, pin_memory=True)
        self.finite = torch.empty((), dtype=torch.bool, pin_memory=True)
        self.generated = torch.cuda.Event(enable_timing=True)
        self.copied = torch.cuda.Event(enable_timing=True)
        self.source = None

    @torch.inference_mode()
    def enqueue(self, image, copy_stream, revision, started_ms):
        if self.source is not None:
            raise RuntimeError("Transfer buffer is still in use")
        self.revision, self.started_ms = revision, started_ms
        compute_stream = torch.cuda.current_stream()
        self.generated.record(compute_stream)
        # A separate stream lets quantization/copy overlap the next model step.
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(self.generated)
            pixels = display_pixels(image).reshape(256, 256)
            finite = torch.isfinite(image).all()
            self.pixels.copy_(pixels, non_blocking=True)
            self.finite.copy_(finite, non_blocking=True)
            self.copied.record(copy_stream)
        # Keep all sources alive until both copies have completed, even on error.
        self.source = image, pixels, finite

    def encode(self):
        if self.source is None:
            raise RuntimeError("No image was enqueued")
        try:
            self.generated.synchronize()
            # Host observation of generation completion, not enqueue time.
            completed_ms = time.time() * 1000
            self.copied.synchronize()
            if not self.finite.item():
                raise FloatingPointError("生成画像にNaNまたはInfが含まれています。")
            start = time.perf_counter()
            png = encode_png(self.pixels.numpy().tobytes())
            return EncodedImage(
                png, self.revision, self.started_ms, completed_ms,
                time.time() * 1000, self.generated.elapsed_time(self.copied),
                (time.perf_counter() - start) * 1000,
            )
        finally:
            self.copied.synchronize()
            self.source = None
