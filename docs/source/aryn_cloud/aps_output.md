# Output Format

The output of the Aryn Partitioning Service is JSON.

```text
{ "status": in-progress updates,
  "error": any errors encountered,
  "elements": a list of elements }
```

## Element Format

{
```text
"type": one of the element types below,
```

Element Types: "N/A","Caption","Footnote","Formula","List-item","Page-footer","Page-header","Image","Section-header","Table","Text","Title"

```text
"bbox": coordinates of the bounding box around the element contents,
```
Takes the format `[x1, y1, x2, y2]` where it is a proportion of how far down the screen the element is.
```text
"properties":
    { "score": confidence (between 0 and 1, with 1 being the most confident),
      "page_number": 1-indexed page number the element occurs on },
```

```text
"text_representation": text associated with this element,
```

Text elements contain ‘\n’ when the text includes a line return

When `extract_images` is set to True, Images include a binary_representation tag which contains a base64 encoded ppm image file of the pdf cropped to the bounds of the detected image. When `extract_images` is false, the bounding box of the Image is still returned.

```text
"binary_representaion": base64 encoded ppm image file of the pdf cropped to the image,
```
}