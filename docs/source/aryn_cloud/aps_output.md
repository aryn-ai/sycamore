# Output Format

The output of the Aryn Partitioning Service is JSON.

```text
{ "status": in-progress updates,
  "error": any errors encountered,
  "elements": a list of elements }
```

## Element Format

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
| Table | A grid of text. See the `extract_table_structure` option to extract information from the table rather than just detecting its presence. |

### Bounding Box

```text
"bbox": coordinates of the bounding box around the element contents
```
Takes the format `[x1, y1, x2, y2]` where each coordinate is a proportion of how far down or across the screen the element is.

### Properties

```text
"properties":
    { "score": confidence (between 0 and 1, with 1 being the most confident 
                that this element type and bounding box coordinates are correct.),
      "page_number": 1-indexed page number the element occurs on }
```

The `score` is the number used to determine whether to keep the element based on the `threshold`. Elements with a confidence score higher than the threshold will be kept as the model is "confident" in its prediction for this element whereas elements with a confidence score lower than this threshold will be removed, as the model is less confident in its prediction for this element.

### Text Representation

```text
"text_representation": text associated with this element
```

Text elements contain ‘\n’ when the text includes a line return.

### Binary Representation

When `extract_images` is set to True, Images include a binary_representation tag which contains a base64 encoded ppm image file of the pdf cropped to the bounds of the detected image. When `extract_images` is false, the bounding box of the Image is still returned.

```text
"binary_representaion": base64 encoded ppm image file of the pdf cropped to the image
```
