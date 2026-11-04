import asyncio
from contextlib import asynccontextmanager
import logging
import time
from urllib.parse import urlsplit

from anyio import CancelScope
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter, ValidationError

from .frames import FrameClient
from .media import decode_image
from .schema import Action, ActionResult, FrameMessage, Snapshot
from .state import Rejected
from .studio import Studio

logger = logging.getLogger(__name__)
frame_message = TypeAdapter(FrameMessage)


def same_origin(connection):
    origin = connection.headers.get("origin")
    if origin is not None and urlsplit(origin).netloc != connection.headers["host"]:
        raise Rejected(403, "別のサイトからは操作できません。")


async def relay(socket, queue):
    async def sending():
        while True:
            value = await queue.get()
            async with asyncio.timeout(5):
                await socket.send_text(value)

    async def receiving():
        while True:
            if await socket.receive_text() != "ping":
                await socket.close(code=1008, reason="Use the HTTP API for operations")
                return

    await run_socket_tasks(sending(), receiving(), False)


async def relay_frames(socket, client):
    send_lock = asyncio.Lock()

    async def sending():
        while True:
            frame = await client.next()
            async with asyncio.timeout(5), send_lock:
                await socket.send_bytes(frame.packet())

    async def receiving():
        while True:
            data = await socket.receive_text()
            received_ms = time.time() * 1000
            try:
                message = frame_message.validate_json(data)
                if message.type == "ack":
                    client.acknowledge(message.id)
                elif message.type == "visibility":
                    client.visibility(message.visible)
                else:
                    async with asyncio.timeout(5), send_lock:
                        await socket.send_json({
                            **message.model_dump(), "received_ms": received_ms,
                            "sent_ms": time.time() * 1000,
                        })
            except (ValidationError, ValueError):
                await socket.close(code=1008, reason="Invalid frame control message")
                return

    await run_socket_tasks(sending(), receiving(), True)


async def run_socket_tasks(sending, receiving, binary):
    tasks = {asyncio.create_task(sending), asyncio.create_task(receiving)}
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect as error:
        if binary and error.code == 4002:
            logger.warning("Client rejected image frame: %r", error.reason)
    except (TimeoutError, OSError, asyncio.CancelledError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        with CancelScope(shield=True):
            await asyncio.gather(*tasks, return_exceptions=True)


def create_app(config, engine_factory):
    studio = Studio(config, engine_factory)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(studio.run())
        try:
            yield
        finally:
            await studio.close(task)

    app = FastAPI(title="Mojika Studio", lifespan=lifespan)
    app.state.studio = studio

    @app.exception_handler(Rejected)
    async def rejected(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    def session(request):
        same_origin(request)
        return studio.require_session(request.cookies.get("mojika_session"))

    @app.post("/api/session", response_model=Snapshot)
    async def initialize(request: Request, response: Response):
        same_origin(request)
        identity = studio.session(request.cookies.get("mojika_session"))
        response.set_cookie(
            "mojika_session",
            identity,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
        )
        return studio.snapshot(identity)

    @app.get("/api/health")
    async def health():
        return {
            "phase": studio.phase,
            "error": studio.error,
            "revision": studio.revision,
            "frames": studio.frame_id,
            "fps": studio.fps,
            "pipeline": studio.pipeline.model_dump(),
            "model": None if studio.info is None else studio.info.model_dump(),
        }

    @app.post("/api/control/acquire", response_model=Snapshot)
    async def acquire(request: Request):
        return await studio.acquire(session(request))

    @app.post("/api/control/release", response_model=Snapshot)
    async def release(request: Request):
        return await studio.release(session(request))

    @app.post("/api/actions", response_model=ActionResult)
    async def action(request: Request, action: Action):
        operation = action.operation
        value = (
            None if operation.kind in ("noise", "clear_conditions") else operation.value
        )
        future = studio.submit(
            session(request), action.request_id, operation.kind, value
        )
        return await asyncio.shield(future)

    @app.post("/api/clip-image", response_model=ActionResult)
    async def clip_image(request: Request, request_id: str):
        identity = session(request)
        studio.lease.require(identity)
        epoch = studio.lease.epoch
        if not 1 <= len(request_id) <= 64:
            raise Rejected(422, "request_idは1〜64文字で指定してください。")
        if request.headers.get("content-type") not in (
            "image/jpeg",
            "image/png",
            "image/webp",
        ):
            raise Rejected(415, "JPEG・PNG・WebPを送信してください。")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 8 * 1024 * 1024:
                raise Rejected(413, "画像は8MB以内にしてください。")
        try:
            image = await asyncio.to_thread(decode_image, bytes(data))
        except ValueError as error:
            raise Rejected(422, str(error)) from error
        # Decoding may outlive a lease handoff. Validate again at submission.
        if studio.lease.epoch != epoch:
            raise Rejected(409, "画像の読込中に操作権が変更されました。")
        return await asyncio.shield(
            studio.submit(identity, request_id, "clip_image", image)
        )

    @app.websocket("/api/events")
    async def events(socket: WebSocket):
        try:
            identity = session(socket)
        except Rejected:
            await socket.close(code=1008)
            return
        if len(studio.events) >= 64:
            await socket.close(code=1013)
            return
        await socket.accept()
        queue = asyncio.Queue(maxsize=1)
        studio.events[queue] = identity
        studio.lease.connect(identity)
        studio.publish()
        try:
            await relay(socket, queue)
        finally:
            del studio.events[queue]
            studio.lease.disconnect(identity, asyncio.get_running_loop().time())
            studio.publish()

    @app.websocket("/api/frames")
    async def frames(socket: WebSocket):
        try:
            session(socket)
        except Rejected:
            await socket.close(code=1008)
            return
        if len(studio.frames) >= 64:
            await socket.close(code=1013)
            return
        await socket.accept()
        client = FrameClient()
        studio.frames.add(client)
        if studio.frame is not None:
            client.offer(studio.frame)
        studio.wake.set()
        studio.publish()
        try:
            await relay_frames(socket, client)
        finally:
            studio.frames.remove(client)
            studio.publish()

    app.mount("/", StaticFiles(directory=config.client_dist, html=True), name="web")
    return app
