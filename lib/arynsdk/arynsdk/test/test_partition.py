import os
from pprint import pprint

from arynsdk.partition import partition_file

# Replace with your token
aryn_token = os.environ["ARYN_API_KEY"]

# Replace with test files
simple_test_file = "/Users/hmlin/Aryn/sycamore/.data/tables/3m_table.pdf"


def test_partition_file():

    f = open(simple_test_file, "rb")
    my_resp = partition_file(f, aryn_token)
    print("----------")
    pprint(my_resp)
    f.close()

    f = open(simple_test_file, "rb")
    my_resp = partition_file(f, aryn_token, use_ocr=True)
    print("----------")
    pprint(my_resp)
    f.close()

    f = open(simple_test_file, "rb")
    my_resp = partition_file(f, aryn_token, threshold=0.35)
    print("----------")
    pprint(my_resp)

    f.seek(0)
    try:
        my_resp = partition_file(f, aryn_token, my_thres=0.4)
        print("----------")
        pprint(my_resp)
    except TypeError as k:
        print(f"Caught incorrect param: {k}")

    f.seek(0)
    my_resp = partition_file(f, aryn_token, extract_table_structure=True, extract_images=True)
    print("----------")
    pprint(my_resp)


def test_table():
    with open(simple_test_file, "rb") as f:
        response = partition_file(f, aryn_token, extract_table_structure=True, use_ocr=True)
    pprint(response)
    for table in filter(lambda e: e["type"] == "table", response["elements"]):
        print(table["dataframe"])


# def test_large_file():

#     f = open(large_test_file, "rb")
#     my_resp = partition_file(f, aryn_token, threshold=0.35)
#     print("----------")
#     pprint(my_resp)

if __name__ == "__main__":
    test_table()
    test_partition_file()
