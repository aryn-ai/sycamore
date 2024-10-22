# Table Extraction From PDF 

In [this example](https://colab.research.google.com/drive/1Qpd-llPC-EPzuTwLfnguMnrQk0eclyqJ?usp=sharing), we’ll use the Partitioning Service to extract the “Supplemental Income” table (shown below) from the 10k financial document  of 3M, and turn it into a pandas dataframe. 

![alt text](3m_supplemental_income.png)

We’ll go through the important code snippets below  to see what’s going on.  (Try it out in  [colab](https://colab.research.google.com/drive/1Qpd-llPC-EPzuTwLfnguMnrQk0eclyqJ?usp=sharing)  yourself! )


Let’s focus on the following code that makes a call to the Aryn Partitioning Service: 

```python
import aryn_sdk
from aryn_sdk.partition import partition_file, tables_to_pandas
import pandas as pd
from io import BytesIO

file = open('my-document.pdf', 'rb')
aryn_api_key = 'YOUR-KEY-HERE'

## Make a call to the Aryn Partitioning Service (APS) 
## param extract_table_structure (boolean): extract tables and their structural content. default: False
## param use_ocr (boolean): extract text using an OCR model instead of extracting embedded text in PDF. default: False
## returns: JSON object with elements representing information inside the PDF
partitioned_file = partition_file(file, aryn_api_key, extract_table_structure=True, use_ocr=True)
```

If you inspect the partitioned_file variable, you’ll notice that it’s a large JSON object with details about all the components in the PDF (checkout [this page](./aps_output.md) to understand the schema of the returned JSON object in detail).  Below, we highlight  the ‘table’ element that contains the information about the table in the page.

```
{'type': 'table',
   'bbox': [0.09080806058995863,
    0.11205035122958097,
    0.8889295869715074,
    0.17521638350053267],
   'properties': {'score': 0.9164711236953735,
    'title': None,
    'columns': None,
    'rows': None,
    'page_number': 1},
   'table': {'cells': [ {'content': '(Millions)',
      'rows': [0],
      'cols': [0],
      'is_header': True,
      'bbox': {'x1': 0.09080806058995863,
       'y1': 0.11341398759321733,
       'x2': 0.40610217823701744,
       'y2': 0.12250489668412642},
      'properties': {}},
     {'content': '2018',
      'rows': [0],
      'cols': [1],
      'is_header': True,
      'bbox': {'x1': 0.6113962958840763,
       'y1': 0.11341398759321733,
       'x2': 0.6766904135311351,
       'y2': 0.12250489668412642},
      'properties': {}},
     {'content': '2017',
      'rows': [0],
      'cols': [2],
      'is_header': True,
      'bbox': {'x1': 0.718455119413488,
       'y1': 0.11341398759321733,
       'x2': 0.7825727664723116,
       'y2': 0.12250489668412642},
      'properties': {}},
     
     ... 
     
     ]}}

```

In particular let's look at the “cells” field  which is an array of cell objects that represent each of the cells in the table. Let’s focus on the first element of that list. 

```
{'cells': [ {'content': '(Millions)',
      'rows': [0],
      'cols': [0],
      'is_header': True,
      'bbox': {'x1': 0.09080806058995863,
       'y1': 0.11341398759321733,
       'x2': 0.40610217823701744,
       'y2': 0.12250489668412642},
      'properties': {}} ... }

```

Here we've detected the first cell, its bounding box (which indicates the coordinates of the cell in the PDF), whether it’s a header cell and its contents. You can then process this JSON however you’d like for further analysis. In [the notebook](https://colab.research.google.com/drive/1Qpd-llPC-EPzuTwLfnguMnrQk0eclyqJ?usp=sharing)  we use the tables_to_pandas function to turn the JSON into a pandas dataframe and then perform some analysis on it:

```python
pandas = tables_to_pandas(partitioned_file)

tables = []
#pull out the tables from the list of elements
for elt, dataframe in pandas:
    if elt['type'] == 'table':
        tables.append(dataframe)
        
supplemental_income = tables[0]
display(supplemental_income)
```

| (Millions) | 2018 | 2017 | 2016 |
| --- | --- | --- | --- |
| Interest expense | 350 | 322 | 199 |
| Interest income | (70) | (50) | (29) |
| Pension and postretirement net periodic benefi... | (73) | (128) | (196) |
| Total | 207 | 144 | (26)  |