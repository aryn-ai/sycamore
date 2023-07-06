from shannon.execution.rules import PushEmbeddingModelConstraint
from shannon.execution.scans import BinaryScan
from shannon.execution.transforms import (
    SentenceTransformerEmbedding, PartitionPDF)


class TestRules:
    def test_push_embedding_model_constraint(self):
        scanner = BinaryScan("s3://bucket/prefix/", binary_format="pdf")
        partitioner = PartitionPDF(scanner, col_name="doc")
        embedder = SentenceTransformerEmbedding(
            partitioner, col_name="doc",
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        rule = PushEmbeddingModelConstraint()
        rule(embedder)
        assert (embedder.batch_size == 3413)
        assert (partitioner.max_partition == 256)
