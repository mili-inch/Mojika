"""Lossless display frames and bounded, per-viewer delivery."""

import asyncio
from dataclasses import dataclass
import io
import struct
import time

from PIL import Image

FRAME_HEADER = struct.Struct("<4sIIddddHHI")
FRAME_WINDOW = 8


def encode_png(pixels):
    if len(pixels) != 256 * 256:
        raise ValueError(f"Invalid image payload: {len(pixels)} bytes; expected 65536")
    output = io.BytesIO()
    Image.frombytes("L", (256, 256), pixels).save(
        output, format="PNG", compress_level=1
    )
    return output.getvalue()


@dataclass(frozen=True)
class EncodedImage:
    png: bytes
    revision: int
    started_ms: float
    completed_ms: float
    encoded_ms: float
    transfer_ms: float
    encode_ms: float


@dataclass(frozen=True)
class Frame:
    id: int
    image: EncodedImage

    def packet(self):
        image = self.image
        return FRAME_HEADER.pack(
            b"MJKP", self.id, image.revision, image.started_ms,
            image.completed_ms, image.encoded_ms, time.time() * 1000,
            256, 256, len(image.png),
        ) + image.png


class FrameClient:
    """Eight unacknowledged frames plus one replaceable latest frame."""

    def __init__(self):
        self.frame = None
        self.inflight = set()
        self.visible = False
        self.changed = asyncio.Event()

    def offer(self, frame):
        self.frame = frame
        self.changed.set()

    def acknowledge(self, frame_id):
        if frame_id not in self.inflight:
            raise ValueError("Acknowledgement does not match an in-flight frame")
        self.inflight.remove(frame_id)
        self.changed.set()

    def visibility(self, visible):
        self.visible = visible
        self.changed.set()

    async def next(self):
        while True:
            await self.changed.wait()
            self.changed.clear()
            if self.visible and len(self.inflight) < FRAME_WINDOW and self.frame:
                frame, self.frame = self.frame, None
                self.inflight.add(frame.id)
                return frame
