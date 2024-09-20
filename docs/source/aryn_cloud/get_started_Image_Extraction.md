# Image Extraction from PDF

In [this example](https://colab.research.google.com/drive/1n5zRm5hfHhxs7dA0FncC44VjlpiPJLWq?usp=sharing), we’ll use the Partitioning Service to extract an image from a battery manual.  We’ll go through the important code snippets below  to see what’s going on (Try it out in  [colab](https://colab.research.google.com/drive/1n5zRm5hfHhxs7dA0FncC44VjlpiPJLWq?usp=sharing)  yourself! )

Let’s focus on the following code that makes a call to the Aryn Partitioning Service: 

```python
import aryn_sdk
from aryn_sdk.partition import partition_file, tables_to_pandas
import pandas as pd
from io import BytesIO

file = open('my-document.pdf', 'rb')
aryn_api_key = 'YOUR-KEY-HERE'

## Make a call to the Aryn Partitioning Service (APS) 
## param use_ocr (boolean): extract text using an OCR model instead of extracting embedded text in PDF. default: False
## param extract_images (boolean):  extract image contents. default: False
## returns: JSON object with elements representing information inside the PDF
partitioned_file = partition_file(file, aryn_api_key, extract_images=True, use_ocr=True)
```

If you inspect the partitioned_file variable, you’ll notice that it’s a large JSON object with details about all the components in the PDF (checkout [this page](./aps_output.md) to understand the schema of the returned JSON object in detail).  Below, we highlight  the ‘Image’ element that contains the information about some of the images in the page: 

```
[
  {
    "type": "Section-header",
    "bbox": [
      0.06470742618336398,
      0.08396875554865056,
      0.3483343505859375,
      0.1039327656139027
    ],
    "properties": {
      "score": 0.7253036499023438,
      "page_number": 1
    },
    "text_representation": "Make AC Power Connections\n"
  },
  {
    "type": "Image",
    "bbox": [
      0.3593270694508272,
      0.10833765896883878,
      0.6269251924402574,
      0.42288088711825284
    ],
    "properties": {
      "score": 0.7996300458908081,
      "image_size": [
        475,
        712
      ],
      "image_mode": "RGB",
      "image_format": null,
      "page_number": 1
    },
    "text_representation": "",
    "binary_representation": "AAAAAA.."
    }, ...
]
```

In particular let's look at the element which highlights the Image that has been detected. 

```
{
    "type": "Image",
    "bbox": [
      0.3593270694508272,
      0.10833765896883878,
      0.6269251924402574,
      0.42288088711825284
    ],
    "properties": {
      "score": 0.7996300458908081,
      "image_size": [
        475,
        712
      ],
      "image_mode": "RGB",
      "image_format": null,
      "page_number": 1
    },
    "text_representation": "",
    "binary_representation": "AAAAAA.."
    }
```

This JSON object represents one of the images in the PDF. You’ll notice that the image’s binary representation, its bounding box (which indicates the coordinates of the image in the PDF), and certain other properties (image_mode, image_size etc.) are returned back. You can then process this JSON however you’d like for further analysis. In the notebook, we use the Pillow Image module from python to display the extracted image on its own. 

```python
## extract all the images from the JSON and print out the JSON representation of the first image
images = [e for e in partitioned_file['elements'] if e['type'] == 'Image']
first_image = images[0]

## read in the image and display it
image_width = first_image['properties']['image_size'][0]
image_height = first_image['properties']['image_size'][1]
image_mode = first_image['properties']['image_mode']
image = Image.frombytes(image_mode,  (image_width, image_height), base64.b64decode(first_image['binary_representation']))

#display the image
image 
```

![alt text](board.png)