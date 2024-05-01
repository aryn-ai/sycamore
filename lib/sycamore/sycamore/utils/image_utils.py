import base64
from io import BytesIO
from typing import Optional

from PIL import Image

from sycamore.data import BoundingBox

DEFAULT_PADDING = 10


def crop_to_bbox(image: Image.Image, bbox: BoundingBox, padding=DEFAULT_PADDING) -> Image.Image:
    """Crops the specified image to the specified bounding box.

    The specified padding is added on all four sides of the box.
    """

    width, height = image.size

    crop_box = (
        bbox.x1 * width - padding,
        bbox.y1 * height - padding,
        bbox.x2 * width + padding,
        bbox.y2 * height + padding,
    )

    cropped_image = image.crop(crop_box)
    return cropped_image


def image_to_bytes(image: Image.Image, format: Optional[str] = None) -> bytes:
    """Converts an image to bytes in the specified format.

    If format is None, returns the raw bytes from the underlying PIL image representation.

    Args:
       image: A PIL image.
       format: The image format to use for serialization. Should be either None or
          a valid format string that can be passed into Image.save().
    """

    if format is None:
        return image.tobytes()

    iobuf = BytesIO()
    image.save(iobuf, format=format)
    return iobuf.getvalue()


def base64_data_url(image: Image.Image) -> str:
    """Returns the image encoded as a png data url

    More info on data urls can be found at https://en.wikipedia.org/wiki/Data_URI_scheme

    Args:
       image: A PIL image.
    """

    encoded_image = image_to_bytes(image, "PNG")
    return f"data:image/png/;base64,{base64.b64encode(encoded_image).decode('utf-8')}"
