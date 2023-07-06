import numpy
import pytest

from shannon.execution.kernels import SentenceTransformerEmbeddingKernel


class TestEmbeddingKernel:

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
    def test_sentence_transformer_embedding_kernel(
            self, model_name, dimension, texts):
        input_batch = {"doc": numpy.array(texts)}
        embedder = SentenceTransformerEmbeddingKernel("doc", model_name)
        output_batch = embedder(doc_batch=input_batch)
        for doc in output_batch["embedding_doc"]:
            assert (len(doc) == dimension)
