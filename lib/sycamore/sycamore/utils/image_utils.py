import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union
from PIL import Image, ImageDraw, ImageFont
from sycamore.data import Document
from sycamore.data.bbox import BoundingBox

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


def image_page_filename_fn(doc: Document) -> str:
    path = Path(doc.properties["path"])
    base_name = ".".join(path.name.split(".")[0:-1])
    page_num = doc.properties["page_number"]
    return f"{base_name}_page_{page_num}.png"


def _is_pair(val, target_type=float):
    return (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and isinstance(val[0], target_type)
        and isinstance(val[1], target_type)
    )


def _default_coord_fn(box) -> tuple[float, float, float, float]:
    """Tries to heuristically extract rectangular coordinates from an object.

    We attempt to pull coordinates from the following formats:

    - BoundingBoxes

    - An object with a 'bbox' attr (like a Document or Element).

    - Lists or tuples of the form:
      - [x_1, y_1, x_2, y_2]
      - [[x_1, y_1], [x_2, y_2]]
      - [[x_1, y_1], [x_3, y_3], [x_2, y_2], [x_4, y_4]]

    - Dictionaries with a key "bbox" and a value that is one of preceding types.

    This particular function does not attempt to distinguish between relative and absolute
    coordinates. That's done in the calling context where we have access to the image size.
    """

    if isinstance(box, BoundingBox):
        return box.coordinates
    elif hasattr(box, "bbox"):
        return _default_coord_fn(box.bbox)
    elif isinstance(box, dict):
        if "bbox" not in box:
            raise ValueError(f"Unable to extract coordinates from dict {box}.")
        return _default_coord_fn(box["bbox"])
    elif isinstance(box, (list, tuple)):
        if len(box) == 2:
            if _is_pair(box[0], float) and _is_pair(box[1], float):
                return (box[0][0], box[0][1], box[1][0], box[1][1])
            raise ValueError(f"Unable to extract coordinates from sequence {box}")

        elif len(box) == 4:
            if all(_is_pair(coord, float) for coord in box):
                return (box[0][0], box[0][1], box[2][0], box[2][1])
            elif all(isinstance(c, float) for c in box):
                return tuple(box)
            raise ValueError(f"Unable to extract coordinates from sequence {box}")

        else:
            raise ValueError(f"Unrecognized list format {box}")
    else:
        raise ValueError(f"Unable to extract coordinates from {box}")


def _default_text_fn(box, index) -> Optional[str]:
    return str(index)


def _default_color_fn(box) -> str:
    return "blue"


U = TypeVar("U", bound=Union[Image.Image, ImageDraw.ImageDraw])


def try_draw_boxes(
    target: U,
    boxes: Any,
    coord_fn: Callable[[Any], tuple[float, float, float, float]] = _default_coord_fn,
    text_fn: Callable[[Any, int], Optional[str]] = _default_text_fn,
    color_fn: Callable[[Any], str] = _default_color_fn,
    font_path: Optional[str] = None,
) -> U:
    """Convenience method to visualize bounding boxes on a PIL image.

    It is often desirable to visualize bounding boxes on a document image to evaluate
    partitioning and OCR techniques. The DrawBoxes function provides this functionality
    for DocSets after partitioning, but over time it has been useful to have this available
    more broadly for testing new libraries or methods.

    This method attempts to "do the right thing" for a variety of bounding box formats and is
    customizable so it can work in a variety of contexts.

    Args:
        target: The Image or ImageDraw instance on which to draw the boxes.
        boxes: Sequence of boxes to draw on the target. The box can be specfiied in a variety of ways,
            and will be interpreted by the methods passed in below.
        coord_fn: Function that takes a box and returns a tuple of coordinates in the form (x1, y1, x2, y2).
            The default attempts to infer the coordinates from a variety of list/dict formats.
        text_fn: Function that takes a box and returns text to display next to each bounding box.
            If text_fn returns None, no text will be rendered.
        color_fn: Function that takes a box and returns the color to draw the box.
        font_path: Path to a TrueType font for rendering text. Deafults to PIL's default font.
    """

    if isinstance(target, ImageDraw.ImageDraw):
        canvas = target
        width, height = target.im.size

    elif isinstance(target, Image.Image):
        canvas = ImageDraw.Draw(target)
        width, height = target.size

    if font_path is not None:
        font = ImageFont.truetype(font_path, 20)
    else:
        font = ImageFont.load_default(size=20)

    for i, box in enumerate(boxes):
        raw_coords = coord_fn(box)

        # If the coordinates are all less than or equal to 1.0, then we treat them
        # as relative coordinates, and we convert them to absolute coordinates.
        if all(c <= 1.0 for c in raw_coords):
            coords = BoundingBox(*raw_coords).to_absolute_self(width, height).coordinates
        else:
            coords = raw_coords

        canvas.rectangle(coords, outline=color_fn(box), width=3)

        text = text_fn(box, i)

        if text is not None:
            text_location = (coords[0] - width / 100, coords[1] - height / 100)
            font_box = canvas.textbbox(text_location, text, font=font)
            canvas.rectangle(font_box, fill="yellow")
            canvas.text(
                text_location,
                text,
                fill="black",
                font=font,
                align="left",
            )

    return target


def show_images(images: Union[Image.Image, list[Image.Image]], width: int = 600) -> None:
    """Displays a list of images for debugging.

    In a Jupyter notebook this uses the display method so that it displays inline.
    In other environments this uses PIL's Image.show(), which opens the image using
    a native file viewer.

    Args:
        images: A PIL image or list of images.
        width: An optional width for the image. This only applies in Jupyter notebooks.
    """
    from IPython.display import display, Image as JImage

    if isinstance(images, Image.Image):
        images = [images]

    # Note: The Jupyter community does not appear to be thrilled with this approach to detect the execution
    # environment, in part because there are a variety of Jupyter frontends, which can be running
    # concurrently. That said, I have not been able to find an alternative that works and provides
    # reasonable behavior in Jupyter, the IPython shell, and a standard Python script/shell.
    # Since this method is intended as a debugging convenience, this seems like a okay place to start.
    # See https://stackoverflow.com/q/15411967 for more context.
    in_jupyter = False

    try:
        ipy_class = get_ipython().__class__.__name__  # type: ignore
        if ipy_class == "ZMQInteractiveShell":
            in_jupyter = True
    except NameError:
        in_jupyter = False

    if in_jupyter:
        for image in images:
            data = BytesIO()
            image.save(data, format="png")
            display(JImage(data=data.getvalue(), width=width))
    else:
        for image in images:
            image.show()
