import io
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch

from mojika.server.transfer import TransferBuffer, display_pixels


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for pinned asynchronous D2H tests")
class FrameTransferTest(unittest.TestCase):
    def test_gpu_quantization_matches_display_including_half_ties(self):
        values = np.concatenate([
            np.linspace(-2, 2, 65000, dtype=np.float32),
            (np.arange(255, dtype=np.float32) + 0.5) / 127.5 - 1,
        ])
        expected = 255 - np.round(np.clip((values + 1) * 127.5, 0, 255)).astype(np.uint8)
        actual = display_pixels(torch.from_numpy(values).cuda()).cpu().numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_double_buffer_event_completion_exact_png_and_reuse(self):
        buffers = [TransferBuffer(), TransferBuffer()]
        pointers = [buffer.pixels.data_ptr() for buffer in buffers]
        stream = torch.cuda.Stream()
        with ThreadPoolExecutor(max_workers=1) as encoder:
            for repeat in range(3):
                expected, jobs = [], []
                for index, buffer in enumerate(buffers):
                    image = torch.linspace(-1, 1, 65536, device="cuda").reshape(256, 256)
                    image = image.roll(repeat * 256 + index * 100)
                    buffer.enqueue(image, stream, index, time.time() * 1000)
                    with self.assertRaisesRegex(RuntimeError, "still in use"):
                        buffer.enqueue(image, stream, 0, 0)
                    jobs.append(encoder.submit(buffer.encode))
                    expected.append(display_pixels(image).cpu().numpy().tobytes())
                for index, job in enumerate(jobs):
                    result = job.result(timeout=5)
                    self.assertEqual(result.revision, index)
                    self.assertGreaterEqual(result.completed_ms, result.started_ms)
                    self.assertGreaterEqual(result.encoded_ms, result.completed_ms)
                    with Image.open(io.BytesIO(result.png)) as png:
                        self.assertEqual(png.mode, "L")
                        self.assertEqual(png.tobytes(), expected[index])
                    self.assertIsNone(buffers[index].source)
        self.assertEqual(pointers, [buffer.pixels.data_ptr() for buffer in buffers])

    def test_nonfinite_is_not_hidden_by_uint8_conversion(self):
        buffer, stream = TransferBuffer(), torch.cuda.Stream()
        for value in (float("nan"), float("inf"), -float("inf")):
            image = torch.zeros((256, 256), device="cuda")
            image[100, 100] = value
            buffer.enqueue(image, stream, 0, time.time() * 1000)
            with self.assertRaises(FloatingPointError):
                buffer.encode()
            self.assertIsNone(buffer.source)
