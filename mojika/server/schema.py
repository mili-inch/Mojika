"""The public studio contract. Exported as JSON Schema for the web client."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


Text = Annotated[str, Field(min_length=1, max_length=4000)]
Pixel = Annotated[StrictInt, Field(ge=0, le=255)]
Hint = Annotated[list[Pixel], Field(min_length=256, max_length=256)]
Interval = Annotated[float, Field(strict=True, ge=0.1, le=60)]
NoiseMix = Annotated[float, Field(strict=True, ge=0, le=1)]


class Controls(Model):
    transition_seconds: Interval
    noise_auto: StrictBool
    noise_mix: NoiseMix

    @classmethod
    def initial(cls):
        return cls(
            transition_seconds=1.0,
            noise_auto=True,
            noise_mix=0.5,
        )


class ClipText(Model):
    kind: Literal["text"]
    text: Text


class ClipImage(Model):
    kind: Literal["image"]
    preview: str


class Conditions(Model):
    gpt: Text | None
    clip: Annotated[ClipText | ClipImage, Field(discriminator="kind")] | None
    hint: Hint | None

    @classmethod
    def initial(cls):
        return cls(gpt=None, clip=None, hint=None)


class TextEdit(Model):
    kind: Literal["gpt", "clip_text"]
    value: Text


class HintEdit(Model):
    kind: Literal["hint"]
    value: Hint


class TransitionEdit(Model):
    kind: Literal["transition_seconds"]
    value: Interval


class NoiseAutoEdit(Model):
    kind: Literal["noise_auto"]
    value: StrictBool


class NoiseMixEdit(Model):
    kind: Literal["noise_mix"]
    value: NoiseMix


class ClearCondition(Model):
    kind: Literal["clear_condition"]
    value: Literal["gpt", "clip", "hint"]


class Button(Model):
    kind: Literal["noise", "clear_conditions"]


Operation = Annotated[
    TextEdit
    | HintEdit
    | TransitionEdit
    | NoiseAutoEdit
    | NoiseMixEdit
    | ClearCondition
    | Button,
    Field(discriminator="kind"),
]


class Action(Model):
    request_id: Annotated[str, Field(min_length=1, max_length=64)]
    operation: Operation


class ActionResult(Model):
    request_id: str
    status: Literal["applied", "superseded", "cancelled", "error"]
    revision: int
    message: str


class ControlLease(Model):
    held: bool
    mine: bool
    reconnecting: bool
    remaining_seconds: float | None


class ModelInfo(Model):
    checkpoint: str
    sha256: str
    device: str
    image_size: int
    steps: int


class PipelineMetrics(Model):
    transfer_ms: float
    encode_ms: float
    frame_bytes: int


class FrameAck(Model):
    type: Literal["ack"]
    id: Annotated[StrictInt, Field(ge=1)]


class FrameVisibility(Model):
    type: Literal["visibility"]
    visible: StrictBool


class FrameClock(Model):
    type: Literal["clock"]
    id: Annotated[StrictInt, Field(ge=0)]
    client_ms: Annotated[float, Field(strict=True, ge=0)]


FrameMessage = Annotated[FrameAck | FrameVisibility | FrameClock, Field(discriminator="type")]


class Snapshot(Model):
    phase: Literal["loading", "warming", "ready", "failed", "stopping"]
    error: str | None
    revision: int
    conditions: Conditions
    controls: Controls
    lease: ControlLease
    viewers: int
    fps: float
    pipeline: PipelineMetrics
    frame_id: int
    frame_revision: int
    pending: int
    model: ModelInfo | None


class Contract(Model):
    action: Action
    result: ActionResult
    snapshot: Snapshot


if __name__ == "__main__":
    import json

    print(json.dumps(Contract.model_json_schema(), ensure_ascii=False, indent=2))
