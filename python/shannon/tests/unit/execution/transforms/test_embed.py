import numpy
import pytest
import ray.data

from shannon.execution import Node
from shannon.execution.transforms import SentenceTransformerEmbedding


class TestEmbedding:
    """Test data is sampled from different captions for the same image from
        the Flickr30k dataset"""

    @pytest.mark.parametrize(
        "model_name, dimension, texts",
        [("sentence-transformers/all-MiniLM-L6-v2", 384,
          ["A group of people are wearing signs that say on strike while"
           " someone is speaking at a booth with the Presidential seal.",
           "Members of a strike at Yale University.",
           "A woman is speaking at a podium outdoors.",
           "A strike is currently going on and there are lots of people.",
           "A person speaks at a protest on a college campus."]),
         ("sentence-transformers/all-MiniLM-L6-v2", 384,
          ["A man laying on the floor as he works on an unfinished wooden"
           " chair with a tool on the ground.",
           "A man putting together a wooden chair.",
           "A man is building a wooden straight-backed chair.",
           "Carpenter building a wooden chair.",
           "A carpenter is fixing up a chair."]),
         ("sentence-transformers/all-mpnet-base-v2", 768,
          ["A large bird stands in the water on the beach.",
           "A white crane stands tall as it looks out upon the ocean.",
           "A gray bird stands majestically on a beach while waves roll in.",
           "A tall bird is standing on the sand beside the ocean.",
           "A water bird standing at the ocean 's edge."])])
    def test_sentence_transformer(
            self, model_name, dimension, texts):
        input_batch = {
            "embedding":
                numpy.array([{"binary": None, "text": None}
                             for i in range(len(texts))]),
            "content":
                numpy.array([{"binary": None, "text": text} for text in texts])
        }

        embedder = SentenceTransformerEmbedding.SentenceTransformer(model_name)
        output_batch = embedder(doc_batch=input_batch)
        for doc in output_batch["embedding"]:
            assert (len(doc["text"]) == dimension)

    def test_sentence_transformer_embedding(self, mocker):
        node = mocker.Mock(spec=Node)
        embedding = SentenceTransformerEmbedding(
            node,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=100)
        input_dataset = ray.data.from_items([
            {
                "doc_id": 1,
                "content": {
                    "binary": None,
                    "text": "Members of a strike at Yale University."},
                "embedding": {
                    "binary": None,
                    "text": None
                }
            },
            {
                "doc_id": 2,
                "content": {
                    "binary": None,
                    "text": "A woman is speaking at a podium outdoors."},
                "embedding": {
                    "binary": None,
                    "text": None
                }
            },
        ])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = embedding.execute()
        output_dataset.show()

    def test_embed(self):
        batch = [None, "str1", "str2"]
        import numpy as np
        ndarray = np.array(batch)
        import pyarrow as pa
        pylist = [{"col": None}, {"col": "str1"}, {"col": "str2"}]
        ptable = pa.Table.from_pylist(pylist)
        batch = [word["col"] for word in ptable.to_pylist()]
        from sentence_transformers import SentenceTransformer
        transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        list = transformer.encode(batch)
        assert (len(list) == 3)
