# Output Format

The output of the Aryn Partitioning Service is JSON.

```text
{ "status": in-progress updates,
  "error": any errors encountered,
  "elements": a list of elements }
```

## Element Format

It is often useful to process different parts of a document separately. For example, you might want to process tables differently than text paragraphs, and typically small chunks of text are embedded separately for vector search. In the Aryn Partitioning Service, these chunks are called elements.

Elements follow the following format:

```text
{"type": type of element,
"bbox": Coordinates of bounding box around element,
"properties": { "score": confidence score,
                "page_number": page number element occurs on},
"text_representation: for elements with associated text,
"binary_representation: for Image elements when extract_table_structure is enabled }
```

An example element is given below:

```json
{
    "type": "Text",
    "bbox": [
      0.10383546717026654,
      0.31373721036044033,
      0.8960905187270221,
      0.39873851429332385
    ],
    "properties": {
      "score": 0.9369918704032898,
      "page_number": 1
    },
    "text_representation": "It is often useful to process different parts of a document separately. For example you\nmight want to process tables differently than text paragraphs, and typically small chunks\nof text are embedded separately for vector search. In the Aryn Partitioning Service, these\nchunks are called elements.\n"
}
```

### Element Type

```text
"type": one of the element types below
```

|  Type   | Description  |
| ------: | :----------- |
| Title   |  Large Text  |
| Text    | Regular Text | 
| Caption | Description of an image or table |
| Footnote | Small text found near the bottom of the page |
| Formula | LaTeX or similar mathematical expression |
| List-item | Part of a list |
| Page-footer | Small text at bottom of page |
| Page-header | Small text at top of page |
| Image | A Picture or diagram. When `extract_images` is set to `true`, this element includes a `binary_representation` tag which contains a base64 encoded ppm image file. When `extract_images` is false, the bounding box of the Image is still returned. |
| Section-header | Medium-sized text marking a section. |
| table | A grid of text. See the `extract_table_structure` option to extract information from the table rather than just detecting its presence. |

### Bounding Box

```text
"bbox": coordinates of the bounding box around the element contents
```
Takes the format `[x1, y1, x2, y2]` where each coordinate is given as the proportion of how far down or across the screen the element is. For instance, an element that is 100 pixels from the left border of a document 400 pixels wide would have an x1 coordinate of 0.25.

### Properties

```text
"properties":
    { "score": confidence (between 0 and 1, with 1 being the most confident 
                that this element type and bounding box coordinates are correct.),
      "page_number": 1-indexed page number the element occurs on }
```

The `score` is the model's "confidence" in its prediction for that particular bounding box. By dafault the model makes its best prediction, but the user can control this using the `threshold` parameter (defaults to "auto"). If the user specifies a numeric value between 0 and 1, only Elements with a confidence score higher than the specified threshold value will be kept. 

### Text Representation

```text
"text_representation": text associated with this element
```

Text elements contain ‘\n’ when the text includes a line return.

### Binary Representation

When `extract_images` is set to True, Images include a `binary_representation` tag which contains a base64 encoded ppm image file of the pdf cropped to the bounds of the detected image. When `extract_images` is false, the bounding box of the Image is still returned.

```text
"binary_representation": base64 encoded ppm image file of the pdf cropped to the image
```
