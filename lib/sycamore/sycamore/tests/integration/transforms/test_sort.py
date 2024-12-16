import sycamore

from sycamore.data import Document


class TestSort:

    def test_sort_docset(self, exec_mode):
        dicts = [
            {
                "doc_id": 1,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that meows"},
                ],
            },
            {
                "doc_id": 2,
                "elements": [
                    {"id": 7, "properties": {"_element_index": 7}, "text_representation": "this is a cat"},
                    {
                        "id": 1,
                        "properties": {"_element_index": 1},
                        "text_representation": "here is an animal that moos",
                    },
                ],
            },
            {
                "doc_id": 3,
                "elements": [
                    {"properties": {"_element_index": 1}, "text_representation": "here is an animal that moos"},
                ],
            },
            {  # handle element with not text
                "doc_id": 4,
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
            {
                "doc_id": 5,
                "elements": [
                    {
                        "properties": {"_element_index": 1},
                        "text_representation": "the number of pages in this document are 253",
                    }
                ],
            },
            {  # drop because of limit
                "doc_id": 6,
                "elements": [
                    {"id": 1, "properties": {"_element_index": 1}},
                ],
            },
        ]
        docs = [Document(item) for item in dicts]

        context = sycamore.init(exec_mode=exec_mode)
        doc_set = context.read.document(docs).sort(descending=True, field="doc_id")
        result = doc_set.take()

        assert len(result) == len(docs)
        assert [doc.doc_id for doc in result] == [6, 5, 4, 3, 2, 1]
