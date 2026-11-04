"""Authoritative shared state, control lease and bounded command mailbox."""

import asyncio
from collections import deque
from dataclasses import dataclass
import time

from .schema import (
    ActionResult,
    ClipImage,
    ClipText,
    Conditions,
    ControlLease,
    Controls,
)


class Rejected(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Lease:
    def __init__(self, grace):
        self.grace = grace
        self.owner = None
        self.deadline = None
        self.epoch = 0
        self.connections = {}

    def connect(self, session):
        self.connections[session] = self.connections.get(session, 0) + 1
        if session == self.owner:
            self.deadline = None

    def disconnect(self, session, now):
        self.connections[session] -= 1
        if self.connections[session] == 0:
            del self.connections[session]
            if self.owner == session:
                self.deadline = now + self.grace

    def acquire(self, session):
        if session not in self.connections:
            raise Rejected(409, "状態への接続が完了してから操作を開始してください。")
        if self.owner is not None and self.owner != session:
            raise Rejected(409, "別のブラウザが操作中です。")
        if self.owner is None:
            self.owner, self.deadline = session, None
            self.epoch += 1

    def require(self, session):
        if self.owner != session or session not in self.connections:
            raise Rejected(403, "操作権を取得してください。")

    def release(self, session):
        self.require(session)
        self.clear()

    def clear(self):
        self.owner, self.deadline = None, None
        self.epoch += 1

    def expire(self, now):
        if self.deadline is not None and now >= self.deadline:
            self.clear()
            return True
        return False

    def view(self, session):
        return ControlLease(
            held=self.owner is not None,
            mine=self.owner == session,
            reconnecting=self.deadline is not None,
            remaining_seconds=None
            if self.deadline is None
            else max(0.0, self.deadline - time.monotonic()),
        )


@dataclass
class Pending:
    request_id: str
    session: str
    epoch: int
    kind: str
    value: object
    future: asyncio.Future

    @property
    def key(self):
        if self.kind in ("noise", "clear_condition", "clear_conditions"):
            return None
        return "clip" if self.kind in ("clip_text", "clip_image") else self.kind


class Mailbox:
    def __init__(self, limit):
        self.limit = limit
        self.items = deque()

    def finish(self, pending, status, revision, message):
        if not pending.future.done():
            pending.future.set_result(
                ActionResult(
                    request_id=pending.request_id,
                    status=status,
                    revision=revision,
                    message=message,
                )
            )

    def put(self, pending, revision):
        # Button actions are ordering barriers for input coalescing.
        if pending.key is not None:
            for previous in reversed(self.items):
                if previous.key is None:
                    break
                if (
                    previous.key == pending.key
                    and previous.session == pending.session
                    and previous.epoch == pending.epoch
                ):
                    self.items.remove(previous)
                    self.finish(
                        previous, "superseded", revision, "新しい入力に置き換えました。"
                    )
                    break
        if len(self.items) >= self.limit:
            raise Rejected(
                429, "操作が混み合っています。少し待ってから操作してください。"
            )
        self.items.append(pending)

    def cancel_all(self, revision, message):
        while self.items:
            self.finish(self.items.popleft(), "cancelled", revision, message)


def changed_state(conditions, controls, kind, value):
    if kind == "clear_conditions":
        return Conditions.initial(), controls
    if kind == "clear_condition":
        return conditions.model_copy(update={value: None}), controls
    if kind == "gpt":
        return conditions.model_copy(update={"gpt": value}), controls
    if kind in ("clip_text", "clip_image"):
        if kind == "clip_image":
            clip = ClipImage(kind="image", preview=value.preview)
        else:
            clip = ClipText(kind="text", text=value)
        return conditions.model_copy(update={"clip": clip}), controls
    if kind == "hint":
        return conditions.model_copy(update={"hint": value}), controls
    if kind == "noise":
        return conditions, controls
    return conditions, Controls.model_validate({**controls.model_dump(), kind: value})
