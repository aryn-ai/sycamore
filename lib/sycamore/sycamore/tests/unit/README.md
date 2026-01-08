DETR partitioner tests use cached results to avoid running inference in during unit test invocations, as the model changes infrequently. 

The current required ground truth files can be generated using the following invocation:

    poetry run python scripts/generate_ground_truth.py --detr-only --file sycamore/tests/resources/data/pdfs/visit_aryn.pdf --file sycamore/tests/resources/data/pdfs/basic_table.pdf --file sycamore/tests/resources/data/pdfs/Ray_page11.pdf --file sycamore/tests/resources/data/pdfs/Ray_page1.pdf
