from pathlib import Path
import tomllib

from pydantic import Field
from .schema import Model


class ServerConfig(Model):
    checkpoint: Path
    gpt_model: Path
    clip_model: Path
    client_dist: Path
    host: str
    port: int = Field(ge=1, le=65535)
    steps: int = Field(ge=1, le=128)
    fps: float = Field(gt=0, le=120)
    seed: int = Field(ge=0)
    compile: bool
    control_grace_seconds: float = Field(ge=0, le=300)

    @classmethod
    def read(cls, path: Path):
        path = path.resolve(strict=True)
        with path.open("rb") as source:
            config = cls.model_validate(tomllib.load(source), strict=False)
        paths = {}
        for field in ("checkpoint", "gpt_model", "clip_model", "client_dist"):
            target = getattr(config, field)
            paths[field] = (path.parent / target).resolve(strict=True)
        config = config.model_copy(update=paths)
        if not (config.client_dist / "index.html").is_file():
            raise ValueError("Build web/ before starting the server.")
        return config
