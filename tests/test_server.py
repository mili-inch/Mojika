import asyncio
import io
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from PIL import Image
from pydantic import ValidationError

from mojika.server.app import create_app
from mojika.server.media import decode_image
from mojika.server.schema import Action, Conditions, Controls, ModelInfo
from mojika.server.state import Lease, Mailbox, Pending, Rejected, changed_state
from mojika.server.studio import Studio
from mojika.server.frames import FRAME_HEADER, FRAME_WINDOW, EncodedImage, FrameClient, encode_png


class FakeBuffer:
    def encode(self):
        start = time.time() * 1000
        png = encode_png(self.pixels)
        return EncodedImage(png, self.revision, self.started_ms, start, time.time() * 1000, 0, 0)


class FakeEngine:
    def __init__(self, config, progress):
        self.buffers = [FakeBuffer(), FakeBuffer()]
        progress("warming")
        self.info = ModelInfo(
            checkpoint="test", sha256="0" * 64, device="test", image_size=256, steps=32
        )

    def apply(self, kind, value):
        if kind == "gpt" and value == "reject":
            raise ValueError("Invalid test prompt")

    def step(self, revision, buffer):
        buffer.pixels = bytes([155]) * 65536
        buffer.revision, buffer.started_ms = revision, time.time() * 1000
        return True


class LeaseTest(unittest.TestCase):
    def test_exclusive_owner_reconnect_and_expiry(self):
        lease = Lease(30)
        lease.connect("a")
        lease.connect("b")
        lease.acquire("a")
        epoch = lease.epoch
        with self.assertRaises(Rejected):
            lease.acquire("b")
        with self.assertRaises(Rejected):
            lease.require("b")
        lease.disconnect("a", 10)
        self.assertFalse(lease.expire(39.9))
        lease.connect("a")
        self.assertEqual(lease.epoch, epoch)
        self.assertFalse(lease.expire(100))
        lease.disconnect("a", 100)
        self.assertTrue(lease.expire(130))
        lease.acquire("b")
        self.assertGreater(lease.epoch, epoch)

    def test_another_tab_keeps_lease_alive(self):
        lease = Lease(30)
        lease.connect("a")
        lease.connect("a")
        lease.acquire("a")
        lease.disconnect("a", 1)
        self.assertFalse(lease.expire(100))
        lease.disconnect("a", 101)
        self.assertTrue(lease.expire(131))


class MailboxTest(unittest.IsolatedAsyncioTestCase):
    def pending(self, kind, identity):
        return Pending(
            identity, "a", 1, kind, None, asyncio.get_running_loop().create_future()
        )

    async def test_latest_input_wins_but_cannot_cross_actions(self):
        mailbox = Mailbox(4)
        first = self.pending("gpt", "1")
        second = self.pending("gpt", "2")
        mailbox.put(first, 0)
        mailbox.put(second, 0)
        self.assertEqual((await first.future).status, "superseded")
        clear = self.pending("clear_conditions", "3")
        mailbox.put(clear, 0)
        third = self.pending("gpt", "4")
        mailbox.put(third, 0)
        self.assertEqual(list(mailbox.items), [second, clear, third])
        mailbox.cancel_all(1, "released")
        self.assertEqual((await second.future).status, "cancelled")
        self.assertEqual((await third.future).status, "cancelled")

    async def test_capacity_and_clip_text_image_share_a_slot(self):
        mailbox = Mailbox(2)
        text = self.pending("clip_text", "1")
        mailbox.put(text, 0)
        image = self.pending("clip_image", "2")
        mailbox.put(image, 0)
        self.assertEqual((await text.future).status, "superseded")
        mailbox.put(self.pending("noise", "3"), 0)
        with self.assertRaises(Rejected):
            mailbox.put(self.pending("noise", "4"), 0)

    async def test_startup_failure_is_visible_and_shutdown_completes(self):
        def fail(config, progress):
            raise RuntimeError("model unavailable")

        studio = Studio(SimpleNamespace(control_grace_seconds=30), fail)
        task = asyncio.create_task(studio.run())
        await task
        self.assertEqual(studio.snapshot("a").phase, "failed")
        self.assertEqual(studio.error, "model unavailable")
        await studio.close(task)

    async def test_invalid_image_payload_fails_without_publishing_a_frame(self):
        class InvalidEngine(FakeEngine):
            def step(self, revision, buffer):
                super().step(revision, buffer)
                buffer.pixels = bytes(65535)
                return True

        studio = Studio(
            SimpleNamespace(control_grace_seconds=30, fps=30), InvalidEngine
        )
        viewer = FrameClient()
        studio.frames.add(viewer)
        with self.assertLogs("mojika.server.studio", level="ERROR"):
            task = asyncio.create_task(studio.run())
            await asyncio.wait_for(task, timeout=2)
        try:
            self.assertEqual(studio.phase, "failed")
            self.assertIn("65535 bytes; expected 65536", studio.error)
            self.assertEqual(studio.frame_id, 0)
            self.assertIsNone(studio.frame)
            self.assertIsNone(viewer.frame)
        finally:
            await studio.close(task)

    async def test_encoding_backlog_is_bounded_and_shutdown_drains_copies(self):
        release = threading.Event()

        class SlowBuffer(FakeBuffer):
            def encode(self):
                if not release.wait(timeout=3):
                    raise TimeoutError("Test did not release encoder")
                return super().encode()

        class SlowEngine(FakeEngine):
            def __init__(self, config, progress):
                super().__init__(config, progress)
                self.buffers = [SlowBuffer(), SlowBuffer()]
                self.calls = 0

            def step(self, revision, buffer):
                self.calls += 1
                return super().step(revision, buffer)

        studio = Studio(SimpleNamespace(control_grace_seconds=30, fps=100), SlowEngine)
        studio.frames.add(FrameClient())
        task = asyncio.create_task(studio.run())
        try:
            async with asyncio.timeout(2):
                while studio.engine is None or studio.engine.calls < 2:
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            self.assertEqual(studio.engine.calls, 2)
            self.assertEqual(studio.frame_id, 0)
            closing = asyncio.create_task(studio.close(task))
            await asyncio.sleep(0.01)
            self.assertFalse(closing.done())
            release.set()
            await asyncio.wait_for(closing, timeout=2)
        finally:
            release.set()
            if not task.done():
                await studio.close(task)


class ContractTest(unittest.TestCase):
    def test_rejects_bad_types_unknown_fields_and_invalid_hint(self):
        for operation in [
            {"kind": "noise_auto", "value": "true"},
            {"kind": "transition_seconds", "value": float("nan")},
            {"kind": "transition_seconds", "value": 0},
            {"kind": "transition_seconds", "value": 0.09},
            {"kind": "noise_mix", "value": -0.01},
            {"kind": "noise_mix", "value": 1.01},
            {"kind": "noise_mix", "value": float("nan")},
            {"kind": "noise_mix", "value": float("inf")},
            {"kind": "noise_mix", "value": "0.5"},
            {"kind": "noise_mix", "value": True},
            {"kind": "hint", "value": [255] * 255},
            {"kind": "hint", "value": [True] * 256},
            {"kind": "gpt", "value": None},
            {"kind": "clip_text", "value": None},
            {"kind": "hint", "value": None},
            {"kind": "gpt", "value": ""},
            {"kind": "clear_conditions", "value": None},
            {"kind": "clear_condition", "value": None},
            {"kind": "clear_condition", "value": "all"},
            {"kind": "clear_condition", "value": "clip_text"},
            {"kind": "noise", "value": ""},
            {"kind": "accuracy", "value": 1},
            {"kind": "unknown", "value": 1},
        ]:
            with self.subTest(operation=operation), self.assertRaises(ValidationError):
                Action.model_validate({"request_id": "test", "operation": operation})

    def test_full_condition_state_and_image_polarity(self):
        conditions, controls = changed_state(
            Conditions.initial(), Controls.initial(), "gpt", "山"
        )
        conditions, controls = changed_state(
            conditions, controls, "hint", [0] + [255] * 255
        )
        conditions, controls = changed_state(conditions, controls, "clip_text", "川")
        self.assertEqual(conditions.gpt, "山")
        self.assertEqual(conditions.hint[0], 0)
        self.assertEqual(conditions.clip.text, "川")
        controls = Controls(
            transition_seconds=3.5,
            noise_auto=False,
            noise_mix=0.4,
        )
        self.assertEqual(
            changed_state(conditions, controls, "clear_conditions", None),
            (Conditions.initial(), controls),
        )
        buffer = io.BytesIO()
        Image.new("L", (32, 32), 127).save(buffer, format="PNG")
        image = decode_image(buffer.getvalue())
        self.assertEqual(image.image.mode, "RGB")
        self.assertTrue(image.preview.startswith("data:image/jpeg;base64,"))
        with self.assertRaises(ValueError):
            decode_image(b"invalid")


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "index.html").write_text("studio")
        self.config = SimpleNamespace(
            client_dist=root, control_grace_seconds=0.1, fps=30
        )
        self.app = create_app(self.config, FakeEngine)
        self.client = TestClient(self.app).__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        for _ in range(100):
            if self.client.get("/api/health").json()["phase"] == "ready":
                break
            time.sleep(0.005)
        self.a = self.new_session()
        self.b = self.new_session()
        self.client.cookies.clear()

    def new_session(self):
        self.client.cookies.clear()
        response = self.client.post("/api/session")
        self.assertEqual(response.status_code, 200)
        return {"cookie": "mojika_session=" + response.cookies["mojika_session"]}

    def action(self, header, kind, value=None, identity="r1"):
        operation = {"kind": kind}
        if kind not in ("noise", "clear_conditions"):
            operation["value"] = value
        return self.client.post(
            "/api/actions",
            headers=header,
            json={"request_id": identity, "operation": operation},
        )

    def test_ownership_complete_sync_handoff_and_binary_frame(self):
        with (
            self.client.websocket_connect("/api/events", headers=self.a) as a,
            self.client.websocket_connect("/api/events", headers=self.b) as b,
        ):
            a.receive_json()
            b.receive_json()
            self.assertEqual(
                self.client.post("/api/control/acquire", headers=self.a).status_code,
                200,
            )
            self.assertEqual(
                self.client.post("/api/control/acquire", headers=self.b).status_code,
                409,
            )
            self.assertEqual(self.action(self.b, "gpt", "bad").status_code, 403)
            response = self.action(self.a, "gpt", "山")
            self.assertEqual(response.json()["status"], "applied")
            while True:
                snapshot = b.receive_json()
                if snapshot["revision"] == 1:
                    break
            self.assertEqual(snapshot["conditions"]["gpt"], "山")
            self.assertFalse(snapshot["lease"]["mine"])
            self.assertEqual(self.action(self.a, "gpt", "duplicate").status_code, 409)
            with self.client.websocket_connect("/api/frames", headers=self.b) as frames:
                frames.send_json({"type": "visibility", "visible": True})
                frame = frames.receive_bytes()
                magic, frame_id, revision, started, completed, encoded, sent, width, height, size = FRAME_HEADER.unpack_from(frame)
                self.assertEqual((magic, revision, width, height), (b"MJKP", 1, 256, 256))
                self.assertGreater(frame_id, 0)
                self.assertGreater(started, 0)
                self.assertLessEqual(started, completed)
                self.assertLessEqual(completed, encoded)
                self.assertLessEqual(encoded, sent)
                self.assertEqual(len(frame), FRAME_HEADER.size + size)
                with Image.open(io.BytesIO(frame[FRAME_HEADER.size:])) as png:
                    self.assertEqual(png.tobytes(), bytes([155]) * 65536)
                frames.send_json({"type": "ack", "id": frame_id})
            self.assertEqual(
                self.client.post("/api/control/release", headers=self.a).status_code,
                200,
            )
            self.assertEqual(
                self.client.post("/api/control/acquire", headers=self.b).status_code,
                200,
            )
            self.assertEqual(
                self.action(self.a, "noise", identity="old").status_code, 403
            )
            self.assertEqual(
                self.action(self.b, "hint", [0] + [255] * 255, identity="hint").json()[
                    "status"
                ],
                "applied",
            )

    def test_frame_rejection_logs_client_diagnostic(self):
        with self.assertLogs("mojika.server.app", level="WARNING") as log:
            with self.client.websocket_connect("/api/frames", headers=self.a) as frames:
                frames.send_json({"type": "visibility", "visible": True})
                frames.receive_bytes()
                frames.close(code=4002, reason="invalid-frame png-decode")
        self.assertIn("invalid-frame png-decode", log.output[0])

    def test_frame_window_keeps_latest_without_blocking_other_viewers(self):
        with self.client.websocket_connect("/api/frames", headers=self.a) as slow:
            slow.send_json({"type": "visibility", "visible": True})
            ids = [FRAME_HEADER.unpack_from(slow.receive_bytes())[1] for _ in range(FRAME_WINDOW)]
            with self.client.websocket_connect("/api/frames", headers=self.b) as fast:
                fast.send_json({"type": "visibility", "visible": True})
                for _ in range(12):
                    frame_id = FRAME_HEADER.unpack_from(fast.receive_bytes())[1]
                    fast.send_json({"type": "ack", "id": frame_id})
                self.assertGreater(frame_id, ids[-1] + 5)
            # Exactly one credit selects the latest image, not the first skipped one.
            slow.send_json({"type": "ack", "id": ids[0]})
            newest = FRAME_HEADER.unpack_from(slow.receive_bytes())[1]
            self.assertGreaterEqual(newest, frame_id)
            slow.send_json({"type": "visibility", "visible": False})
            for frame_id in ids[1:] + [newest]:
                slow.send_json({"type": "ack", "id": frame_id})
            slow.send_json({"type": "clock", "id": 1, "client_ms": 1234.5})
            clock = slow.receive_json()
            self.assertEqual(clock["client_ms"], 1234.5)
            self.assertGreaterEqual(clock["sent_ms"], clock["received_ms"])
            time.sleep(0.2)
            self.assertTrue(all(not client.inflight for client in self.app.state.studio.frames))
            slow.send_json({"type": "visibility", "visible": True})
            self.assertGreater(FRAME_HEADER.unpack_from(slow.receive_bytes())[1], newest + 2)

    def test_rejects_invalid_frame_controls(self):
        for message in [
            {"type": "ack", "id": 1}, {"type": "ack", "id": True},
            {"type": "visibility", "visible": "true"},
            {"type": "clock", "id": 0, "client_ms": float("nan")},
            {"type": "visibility", "visible": True, "extra": 0},
        ]:
            with self.subTest(message=message), self.client.websocket_connect("/api/frames", headers=self.a) as frames:
                frames.send_json(message)
                with self.assertRaises(WebSocketDisconnect) as error:
                    frames.receive_bytes()
                self.assertEqual(error.exception.code, 1008)

    def test_reconnect_restores_state_and_lease_then_expires(self):
        with self.client.websocket_connect("/api/events", headers=self.a):
            self.client.post("/api/control/acquire", headers=self.a)
            self.action(self.a, "transition_seconds", 2.5)
            self.action(self.a, "noise_mix", 0.4, "mix")
        with self.client.websocket_connect("/api/events", headers=self.a) as resumed:
            snapshot = resumed.receive_json()
            self.assertTrue(snapshot["lease"]["mine"])
            self.assertEqual(snapshot["controls"]["transition_seconds"], 2.5)
            self.assertEqual(snapshot["controls"]["noise_mix"], 0.4)
        time.sleep(0.35)
        with self.client.websocket_connect("/api/events", headers=self.b):
            self.assertEqual(
                self.client.post("/api/control/acquire", headers=self.b).status_code,
                200,
            )

    def test_individual_clear_syncs_only_selected_condition(self):
        with self.client.websocket_connect("/api/events", headers=self.a) as events:
            self.client.post("/api/control/acquire", headers=self.a)
            for i, (kind, value) in enumerate(
                (("gpt", "山"), ("clip_text", "木"), ("hint", [0] * 256))
            ):
                self.action(self.a, kind, value, str(i))
            expected = {
                "gpt": "山",
                "clip": {"kind": "text", "text": "木"},
                "hint": [0] * 256,
            }
            for revision, selected in enumerate(("clip", "hint", "gpt"), start=4):
                result = self.action(
                    self.a, "clear_condition", selected, "clear-" + selected
                ).json()
                self.assertEqual(result["status"], "applied")
                while True:
                    snapshot = events.receive_json()
                    if snapshot["revision"] == revision:
                        break
                expected[selected] = None
                self.assertEqual(snapshot["conditions"], expected)
                self.assertEqual(snapshot["controls"], Controls.initial().model_dump())

    def test_clear_conditions_is_one_shared_revision_and_preserves_controls(self):
        with self.client.websocket_connect("/api/events", headers=self.a) as events:
            self.client.post("/api/control/acquire", headers=self.a)
            for i, (kind, value) in enumerate(
                (
                    ("gpt", "山"),
                    ("clip_text", "木"),
                    ("hint", [255] * 255 + [0]),
                    ("transition_seconds", 2.5),
                    ("noise_auto", False),
                    ("noise_mix", 0.4),
                )
            ):
                self.assertEqual(
                    self.action(self.a, kind, value, str(i)).json()["status"], "applied"
                )
            response = self.action(self.a, "clear_conditions", identity="clear")
            self.assertEqual(response.json()["revision"], 7)
            while True:
                snapshot = events.receive_json()
                if snapshot["revision"] == 7:
                    break
            self.assertEqual(
                snapshot["conditions"], {"gpt": None, "clip": None, "hint": None}
            )
            self.assertEqual(
                snapshot["controls"],
                {
                    "transition_seconds": 2.5,
                    "noise_auto": False,
                    "noise_mix": 0.4,
                },
            )

    def test_noise_mix_syncs_to_viewers_and_rejects_invalid_input(self):
        with (
            self.client.websocket_connect("/api/events", headers=self.a),
            self.client.websocket_connect("/api/events", headers=self.b) as viewer,
        ):
            self.assertEqual(viewer.receive_json()["controls"]["noise_mix"], 0.5)
            self.client.post("/api/control/acquire", headers=self.a)
            self.assertEqual(self.action(self.b, "noise_mix", 0.3).status_code, 403)
            for revision, mix in enumerate((0, 0.4, 1), start=1):
                result = self.action(self.a, "noise_mix", mix, str(revision)).json()
                self.assertEqual(result["status"], "applied")
                while True:
                    snapshot = viewer.receive_json()
                    if snapshot["revision"] == revision:
                        break
                self.assertEqual(snapshot["controls"]["noise_mix"], mix)
            self.assertEqual(
                self.action(self.a, "noise_mix", 1.1, "invalid").status_code, 422
            )
            self.assertEqual(self.client.get("/api/health").json()["revision"], 3)


    def test_explicit_image_endpoint_and_rejected_input_do_not_change_state(self):
        with self.client.websocket_connect("/api/events", headers=self.a):
            self.client.post("/api/control/acquire", headers=self.a)
            self.assertEqual(
                self.action(self.a, "gpt", "reject").json()["status"], "error"
            )
            self.assertEqual(self.client.get("/api/health").json()["revision"], 0)
            buffer = io.BytesIO()
            Image.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
            response = self.client.post(
                "/api/clip-image?request_id=photo",
                headers={**self.a, "content-type": "image/png"},
                content=buffer.getvalue(),
            )
            self.assertEqual(response.json()["status"], "applied")
            self.assertEqual(
                self.client.post(
                    "/api/clip-image?request_id=bad",
                    headers={**self.a, "content-type": "image/png"},
                    content=b"bad",
                ).status_code,
                422,
            )
            self.assertEqual(
                self.client.post(
                    "/api/actions",
                    headers={**self.a, "origin": "https://wrong.example"},
                    json={"request_id": "x", "operation": {"kind": "noise"}},
                ).status_code,
                403,
            )
            self.assertEqual(self.client.get("/health").status_code, 404)
