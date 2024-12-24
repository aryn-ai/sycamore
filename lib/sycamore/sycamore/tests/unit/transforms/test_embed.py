import pickle
import pytest
import ray.data

from sycamore.data import Document, Element
from sycamore.plan_nodes import Node
from sycamore.transforms import Embed
from sycamore.transforms.embed import OpenAIEmbedder, SentenceTransformerEmbedder, BedrockEmbedder


class TestEmbedding:

    def check_sentence_transformer(
        self, model_name, dimension, texts, use_documents: bool = False, use_elements: bool = False
    ):
        input_batch = []
        for text in texts:
            doc = Document()
            if use_documents:
                doc.text_representation = text
            if use_elements:
                element = Element()
                element.text_representation = text
                doc.elements = [element]
            input_batch.append(doc)
        embedder = SentenceTransformerEmbedder(model_name)
        output_batch = embedder(doc_batch=input_batch)
        for doc in output_batch:
            if use_documents:
                assert doc.embedding is not None
                assert len(doc.embedding) == dimension
            if use_elements:
                for element in doc.elements:
                    assert element.embedding is not None
                    assert len(element.embedding) == dimension

    """Test data is sampled from different captions for the same image from
    the Flickr30k dataset"""

    @pytest.mark.parametrize(
        "model_name, dimension, texts",
        [
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                384,
                [
                    "A group of people are wearing signs that say on strike while"
                    " someone is speaking at a booth with the Presidential seal.",
                    "Members of a strike at Yale University.",
                    "A woman is speaking at a podium outdoors.",
                    "A strike is currently going on and there are lots of people.",
                    "A person speaks at a protest on a college campus.",
                ],
            ),
            (
                "sentence-transformers/all-MiniLM-L6-v2",
                384,
                [
                    "A man laying on the floor as he works on an unfinished wooden" " chair with a tool on the ground.",
                    "A man putting together a wooden chair.",
                    "A man is building a wooden straight-backed chair.",
                    "Carpenter building a wooden chair.",
                    "A carpenter is fixing up a chair.",
                ],
            ),
            (
                "sentence-transformers/all-mpnet-base-v2",
                768,
                [
                    "A large bird stands in the water on the beach.",
                    "A white crane stands tall as it looks out upon the ocean.",
                    "A gray bird stands majestically on a beach while waves roll in.",
                    "A tall bird is standing on the sand beside the ocean.",
                    "A water bird standing at the ocean 's edge.",
                ],
            ),
        ],
    )
    def test_sentence_transformer(self, model_name, dimension, texts):
        self.check_sentence_transformer(model_name, dimension, texts, use_documents=True, use_elements=True)
        self.check_sentence_transformer(model_name, dimension, texts, use_elements=True)
        self.check_sentence_transformer(model_name, dimension, texts, use_documents=True)

    def check_sentence_transformer_embedding(self, mocker, use_documents: bool = False, use_elements: bool = False):
        node = mocker.Mock(spec=Node)
        embedding = Embed(
            node,
            embedder=SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=100),
        )
        texts = ["Members of a strike at Yale University.", "A woman is speaking at a podium outdoors."]
        elements = [
            {"_element_index": 1, "text_representation": texts[0], "embedding": None},
            {
                "_element_index": 2,
                "text_representation": texts[1],
                "embedding": None,
            },
        ]
        dicts = [
            {
                "doc_id": 1,
                "text_representation": texts[0] if use_documents else None,
                "embedding": None,
                "elements": elements if use_elements else [],
            },
            {"doc_id": 2, "text_representation": texts[1] if use_documents else None, "embedding": None},
        ]
        input_dataset = ray.data.from_items([{"doc": Document(dict).serialize()} for dict in dicts])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        input_dataset.show()
        output_dataset = embedding.execute()
        output_dataset.show()

    def test_sentence_transformer_embedding(self, mocker):
        self.check_sentence_transformer_embedding(mocker, use_documents=True, use_elements=True)
        self.check_sentence_transformer_embedding(mocker, use_elements=True)
        self.check_sentence_transformer_embedding(mocker, use_documents=True)

    def test_openai_embedder_pickle(self):
        obj = OpenAIEmbedder()
        obj._client = obj.client_wrapper.get_client()

        pickle.dumps(obj)
        assert True

    def test_sentence_transformer_batch_size(self):
        embedder = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert embedder.model_batch_size == 100

        embedder = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", model_batch_size=50)
        assert embedder.model_batch_size == 50

        embedder = BedrockEmbedder(model_batch_size=100)
        assert embedder.model_batch_size == 1

        embedder = BedrockEmbedder(model_batch_size=1)
        assert embedder.model_batch_size == 1

        embedder = OpenAIEmbedder(model_batch_size=120)
        assert embedder.model_batch_size == 120

        # Test batching using SentenceTransformer
        texts = ["text1", "text2", "text3", "text4"]
        docs = [Document({"text_representation": t}) for t in texts]

        embedders = [
            SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", model_batch_size=2),
            OpenAIEmbedder(model_batch_size=2),
            BedrockEmbedder(model_batch_size=1),
        ]
        for embedder in embedders:
            original_embed_texts = embedder.embed_texts

            def mock_embed_texts(text_batch):
                assert len(text_batch) <= 2, "All batches should be size 2 or smaller"
                return original_embed_texts(text_batch)

            embedder.embed_texts = mock_embed_texts

            embedded_docs = embedder(docs)
            assert len(embedded_docs) == len(texts), "All texts should be processed"
