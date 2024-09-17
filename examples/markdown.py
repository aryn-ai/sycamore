import sys

import sycamore
from sycamore.transforms.partition import ArynPartitioner

docs = (
    sycamore.init()
    .read.binary(sys.argv[1:], binary_format="pdf")
    .partition(partitioner=ArynPartitioner(extract_table_structure=True,
                                           use_partitioning_service=False))
    .markdown()
    .explode()
    .take()
)

print(docs[1].text_representation)  # doc zero isn't real
