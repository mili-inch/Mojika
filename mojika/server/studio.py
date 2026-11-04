"""Event-loop state, one inference worker, and bounded transfer/encoding jobs."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import secrets
import time

from .frames import FRAME_HEADER, Frame
from .schema import Conditions, Controls, PipelineMetrics, Snapshot
from .state import Lease, Mailbox, Pending, Rejected, changed_state

logger = logging.getLogger(__name__)


def latest(queue, value):
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(value)


class Studio:
    def __init__(self, config, engine_factory):
        self.config, self.engine_factory = config, engine_factory
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mojika-gpu"
        )
        self.encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mojika-png")
        self.phase, self.error, self.info = "loading", None, None
        self.conditions, self.controls = Conditions.initial(), Controls.initial()
        self.revision = self.frame_id = self.frame_revision = 0
        self.fps = 0.0
        self.pipeline = PipelineMetrics(transfer_ms=0, encode_ms=0, frame_bytes=0)
        self.frame = None
        self.lease = Lease(config.control_grace_seconds)
        self.mailbox = Mailbox(32)
        self.sessions = {}
        self.events = {}
        self.frames = set()
        self.requests = {}
        self.transition = asyncio.Lock()
        self.wake = asyncio.Event()
        self.stopping = False
        self.engine = None

    def session(self, cookie):
        now = time.monotonic()
        for session, seen in list(self.sessions.items()):
            if (
                now - seen > 3600
                and session not in self.lease.connections
                and session != self.lease.owner
            ):
                del self.sessions[session]
        if cookie in self.sessions:
            self.sessions[cookie] = now
            return cookie
        if len(self.sessions) >= 256:
            raise Rejected(429, "接続数の上限に達しています。")
        session = secrets.token_urlsafe(32)
        self.sessions[session] = now
        return session

    def require_session(self, cookie):
        if cookie not in self.sessions:
            raise Rejected(401, "接続を初期化してください。")
        self.sessions[cookie] = time.monotonic()
        return cookie

    def snapshot(self, session):
        return Snapshot(
            phase=self.phase,
            error=self.error,
            revision=self.revision,
            conditions=self.conditions,
            controls=self.controls,
            lease=self.lease.view(session),
            viewers=len(self.frames),
            fps=round(self.fps, 1),
            pipeline=self.pipeline,
            frame_id=self.frame_id,
            frame_revision=self.frame_revision,
            pending=len(self.mailbox.items),
            model=self.info,
        )

    def publish(self):
        for queue, session in self.events.items():
            latest(queue, self.snapshot(session).model_dump_json())

    def phase_changed(self, phase):
        self.phase = phase
        self.publish()

    async def call(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, function, *args
        )

    async def acquire(self, session):
        async with self.transition:
            self.lease.acquire(session)
            self.publish()
        return self.snapshot(session)

    async def release(self, session):
        async with self.transition:
            self.lease.release(session)
            self.mailbox.cancel_all(self.revision, "操作権を解放しました。")
            self.publish()
        return self.snapshot(session)

    def submit(self, session, request_id, kind, value):
        self.lease.require(session)
        if self.phase != "ready":
            raise Rejected(503, "生成器が準備できていません。")
        key = session, request_id
        if key in self.requests:
            raise Rejected(409, "request_idが重複しています。")
        future = asyncio.get_running_loop().create_future()
        pending = Pending(request_id, session, self.lease.epoch, kind, value, future)
        self.mailbox.put(pending, self.revision)
        self.requests[key] = True
        if len(self.requests) > 512:
            del self.requests[next(iter(self.requests))]
        self.wake.set()
        self.publish()
        return future

    async def run(self):
        loop = asyncio.get_running_loop()
        active = None
        pending = {}
        try:
            self.engine = await self.call(
                self.engine_factory,
                self.config,
                lambda phase: loop.call_soon_threadsafe(self.phase_changed, phase),
            )
            self.info = self.engine.info
            available = list(self.engine.buffers)
            self.phase_changed("ready")
            measured_at, measured_frames = time.monotonic(), self.frame_id
            next_step = measured_at
            while not self.stopping:
                start = time.monotonic()
                self.wake.clear()
                for future in list(pending):
                    if future.done():
                        image = future.result()
                        available.append(pending.pop(future))
                        self.frame_id += 1
                        self.frame_revision = image.revision
                        self.frame = Frame(self.frame_id, image)
                        self.pipeline = PipelineMetrics(
                            transfer_ms=image.transfer_ms, encode_ms=image.encode_ms,
                            frame_bytes=len(image.png) + FRAME_HEADER.size,
                        )
                        for client in self.frames:
                            client.offer(self.frame)
                async with self.transition:
                    if self.lease.expire(start):
                        self.mailbox.cancel_all(
                            self.revision, "操作権の接続待ち時間が終了しました。"
                        )
                        self.publish()
                    if self.mailbox.items:
                        active = self.mailbox.items.popleft()
                        if (
                            active.session != self.lease.owner
                            or active.epoch != self.lease.epoch
                        ):
                            self.mailbox.finish(
                                active,
                                "cancelled",
                                self.revision,
                                "操作権が変更されました。",
                            )
                        else:
                            try:
                                await self.call(
                                    self.engine.apply,
                                    active.kind,
                                    active.value,
                                )
                            except ValueError as error:
                                self.mailbox.finish(
                                    active, "error", self.revision, str(error)
                                )
                            else:
                                self.conditions, self.controls = changed_state(
                                    self.conditions,
                                    self.controls,
                                    active.kind,
                                    active.value,
                                )
                                self.revision += 1
                                self.mailbox.finish(
                                    active, "applied", self.revision, "適用しました。"
                                )
                        active = None
                        self.publish()
                if self.frames and available and time.monotonic() >= next_step:
                    step_start = time.monotonic()
                    buffer = available.pop()
                    if await self.call(self.engine.step, self.revision, buffer):
                        future = loop.run_in_executor(self.encoder, buffer.encode)
                        pending[future] = buffer
                        future.add_done_callback(lambda _: self.wake.set())
                    else:
                        available.append(buffer)
                    next_step = step_start + 1 / self.config.fps
                now = time.monotonic()
                if now - measured_at >= 1:
                    self.fps = (self.frame_id - measured_frames) / (now - measured_at)
                    measured_at, measured_frames = now, self.frame_id
                    self.publish()
                timeout = (
                    max(0.001, next_step - now)
                    if self.frames and available
                    else 0.25
                )
                if self.mailbox.items:
                    continue
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout)
                except TimeoutError:
                    pass
        except Exception as error:
            logger.exception("Generation engine failed")
            self.error, self.phase = str(error), "failed"
            if active is not None:
                self.mailbox.finish(
                    active, "error", self.revision, "生成器が停止しました。"
                )
            self.mailbox.cancel_all(self.revision, "生成器が停止しました。")
            self.publish()
        finally:
            # Encoder jobs retain pinned/GPU buffers until their copy events finish.
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self, task):
        self.stopping = True
        self.phase_changed("stopping")
        self.wake.set()
        await task
        self.mailbox.cancel_all(self.revision, "サーバーを停止しました。")
        await asyncio.to_thread(self.executor.shutdown, wait=True)
        await asyncio.to_thread(self.encoder.shutdown, wait=True)
