from transformers import AutoTokenizer

from sycamore.data import Document
from sycamore.transforms.merge_elements import GreedyElementMerger


class TestMergeElements:
    dict0 = {
        "doc_id": "doc_id",
        "type": "pdf",
        "text_representation": "text",
        "binary_representation": None,
        "parent_id": None,
        "properties": {"path": "/docs/foo.txt", "title": "bar"},
        "elements": {
            "array": [
                {
                    "type": "UncategorizedText",
                    "text_representation": "text1",
                    "properties": {"filetype": "text/plain", "page_number": 1},
                },
                {
                    "type": "UncategorizedText",
                    "text_representation": "text2",
                    "properties": {"filetype": "text/plain", "page_number": 2},
                },
                {
                    "type": "Title",
                    "text_representation": "title1",
                    "bbox": [0.17606372549019608, 0.4045761636363636, 0.8238071275686271, 0.47917035555555554],
                    "binary_representation": b"title1",
                    "properties": {"doc_title": "title"},
                },
                {
                    "type": "NarrativeText",
                    "text_representation": """Recurrent neural networks, long short-term memory [12]
                                            and gated recurrent [7] neural networks in particular,
                                            have been ﬁrmly established as state of the art approaches
                                            in sequence modeling and transduction problems such as
                                            language modeling and machine translation [29, 2, 5].
                                            Numerous efforts have since continued to push the boundaries
                                            of recurrent language models and encoder-decoder architectures
                                            [31, 21, 13].
                                            ∗Equal contribution. Listing order is random. Jakob proposed
                                            replacing RNNs with self-attention and started the effort to
                                            evaluate this idea. Ashish, with Illia, designed and implemented
                                            the ﬁrst Transformer models and has been crucially involved
                                            in every aspect of this work. Noam proposed scaled dot-product
                                            attention, multi-head attention and the parameter-free position
                                            representation and became the other person involved in nearly
                                            every detail. Niki designed, implemented, tuned and evaluated
                                            countless model variants in our original codebase and tensor2tensor.
                                            Llion also experimented with novel model variants, was responsible
                                            for our initial codebase, and efﬁcient inference and visualizations.
                                            Lukasz and Aidan spent countless long days designing various parts
                                            of and implementing tensor2tensor, replacing our earlier codebase,
                                            greatly improving results and massively accelerating our research.
                                            †Work performed while at Google Brain. ‡Work performed while at
                                            Google Research.
                                            31st Conference on Neural Information Processing Systems
                                            (NIPS 2017), Long Beach, CA, USA.""",
                    "bbox": [0.27606372549019608, 0.5045761636363636, 0.9238071275686271, 0.67917035555555554],
                    "properties": {"doc_title": "other title", "prop2": "prop 2 value"},
                },
                {
                    "type": "NarrativeText",
                    "text_representation": """Recurrent neural networks, long short-term memory [12]
                                            and gated recurrent [7] neural networks in particular,
                                            have been ﬁrmly established as state of the art approaches
                                            in sequence modeling and transduction problems such as
                                            language modeling and machine translation [29, 2, 5].
                                            Numerous efforts have since continued to push the boundaries
                                            of recurrent language models and encoder-decoder architectures
                                            [31, 21, 13].
                                            ∗Equal contribution. Listing order is random. Jakob proposed
                                            replacing RNNs with self-attention and started the effort to
                                            evaluate this idea. Ashish, with Illia, designed and implemented
                                            the ﬁrst Transformer models and has been crucially involved
                                            in every aspect of this work. Noam proposed scaled dot-product
                                            attention, multi-head attention and the parameter-free position
                                            representation and became the other person involved in nearly
                                            every detail. Niki designed, implemented, tuned and evaluated
                                            countless model variants in our original codebase and tensor2tensor.
                                            Llion also experimented with novel model variants, was responsible
                                            for our initial codebase, and efﬁcient inference and visualizations.
                                            Lukasz and Aidan spent countless long days designing various parts
                                            of and implementing tensor2tensor, replacing our earlier codebase,
                                            greatly improving results and massively accelerating our research.
                                            †Work performed while at Google Brain. ‡Work performed while at
                                            Google Research.
                                            31st Conference on Neural Information Processing Systems
                                            (NIPS 2017), Long Beach, CA, USA.""",
                    "bbox": [0.27606372549019608, 0.5045761636363636, 0.9238071275686271, 0.67917035555555554],
                    "properties": {"doc_title": "other title", "prop2": "prop 2 value"},
                },
                {
                    "type": "Title",
                    "text_representation": "title1",
                    "bbox": [0.17606372549019608, 0.4045761636363636, 0.8238071275686271, 0.47917035555555554],
                    "binary_representation": b"title1",
                    "properties": {"doc_title": "title"},
                },
                {},
            ]
        },
    }

    def test_merge_elements(self):
        doc = Document(self.dict0)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        merger = GreedyElementMerger(tokenizer, 512)
        new_doc = merger.merge_elements(doc)
        assert len(new_doc.elements) == 4

        e = new_doc.elements[0]
        assert e.type == "Section"
        assert e.text_representation == "text1\ntext2"
        assert e.properties == {"filetype": "text/plain", "page_number": 1}

        e = new_doc.elements[1]
        assert e.type == "Section"
        assert e.bbox.coordinates == (0.17606372549019608, 0.4045761636363636, 0.9238071275686271, 0.67917035555555554)
        assert e.properties == {"doc_title": "title", "prop2": "prop 2 value"}

        e = new_doc.elements[2]
        assert e.type == "NarrativeText"

        e = new_doc.elements[3]
        assert e.type == "Section"
        assert e.text_representation == "title1"
        assert e.bbox.coordinates == (0.17606372549019608, 0.4045761636363636, 0.8238071275686271, 0.47917035555555554)
        assert e.binary_representation == b"title1"
        assert e.properties == {"doc_title": "title"}
