import base64
from dataclasses import dataclass
import io

from PIL import Image, UnidentifiedImageError, ImageOps


@dataclass(frozen=True)
class ImageInput:
    image: Image.Image
    preview: str


def decode_image(data: bytes) -> ImageInput:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("画像は1,600万画素以内にしてください。")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("画像を読み込めません。") from error
    preview = image.copy()
    preview.thumbnail((224, 224))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    return ImageInput(
        image, "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    )
