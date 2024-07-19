import io
import requests
import json

# Replace with your token
aryn_token = "YOUR_TOKEN"

# Replace with test files
simple_test_file = "PATH_TO_SIMPLE_FILE"
large_test_file = "PATH_TO_LARGE_FILE"

# URL for Aryn Partitioning Service (APS)
aps_url = "https://api.aryn.cloud/v1/document/partition"

def check_options(kwargs: dict):

    valid_options = ("threshold", "use_ocr", "extract_table_structure", "extract_images")

    for key in kwargs:
        if key not in valid_options:
            raise KeyError(f"{key} is not a valid option")

    return True

#
# Sends file to the Aryn Partitioning Service and returns a dict of its document structure and text
# 
#
# Options for the Aryn Partitioning Service are:
#
#        threshold:  value in [0.0 .. 1.0] to specify the cutoff for detecting bounding boxes.
#                    default: 0.4
#        use_ocr:    boolean to specify extracting text using an OCR model instead of extracting embedded text in PDF.
#                    default: False
#        extract_table_structure: boolean to specfy extracting tables and their structural content.
#                    default: False
#        extract_images: boolean that Mark doesn't know what it does.
#                    default: False
#
#        The defaults are what the Service will use, if not passed into the function

def partition_file(file: io, token: str, **kwargs) -> dict:

    check_options(kwargs)

    options_str = json.dumps(kwargs)

    print(f"{options_str}")

    files = { "options": options_str.encode('utf-8'),
              "pdf": file }
    
    http_header = {"Authorization": "Bearer {}".format(token)}

    files = { "pdf": file }
    
    resp = requests.post(aps_url, files=files, headers=http_header)

    if resp.status_code != 200:
        raise requests.exceptions.HTTPError(f"Error: status_code: {resp.status_code}, reason: {resp.text}")

    return resp.json()


def add_bbox_to_pdf():

    return "Unimplemented"


def test_partition_file():

    f = open(simple_test_file, "rb")
    my_resp = partition_file(f, aryn_token)
    print("----------")
    print(my_resp)
    f.close()

    f = open(simple_test_file, "rb")
    my_resp = partition_file(f, aryn_token)
    print("----------")
    print(my_resp)
    f.close()

    f = open(simple_test_file, "rb")
    my_resp = partition_file(f, aryn_token, threshold = 0.35)
    print("----------")
    print(my_resp)

    f.seek(0)
    try:
        my_resp = partition_file(f, aryn_token, my_thres = 0.4)
        print("----------")
        print(my_resp)
    except KeyError as k:
        print(f"Caught incorrect param: {k}")

    f.seek(0)
    my_resp = partition_file(f, aryn_token, extract_table_structure = True, extract_images = True)
    print("----------")
    print(my_resp)


def test_large_file():

    f = open(large_test_file, "rb")
    my_resp = partition_file(f, aryn_token, threshold = 0.35)
    print("----------")
    print(my_resp)


#test_partition_file()
#test_large_file()
