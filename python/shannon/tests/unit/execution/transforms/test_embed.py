import ray.data
from shannon.execution import Node
from shannon.execution.transforms import SentenceTransformerEmbedding


class TestEmbedding:
    def test_sentence_transformer_embedding(self, mocker):
        node = mocker.Mock(spec=Node)
        embedding = SentenceTransformerEmbedding(
            node,
            col_name="doc",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=100)
        input_dataset = ray.data.from_items([
            {"index": 1, "doc": "Members of a strike at Yale University."},
            {"index": 2, "doc": "A woman is speaking at a podium outdoors."}
        ])
        execute = mocker.patch.object(node, "execute")
        execute.return_value = input_dataset
        output_dataset = embedding.execute()
        output_dataset.show()
