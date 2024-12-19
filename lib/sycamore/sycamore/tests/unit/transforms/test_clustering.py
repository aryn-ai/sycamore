import pytest
import ray.data

import sycamore
from sycamore import DocSet
from sycamore.data import Document
from sycamore.transforms.clustering import KMeans


class TestKMeans:
    @pytest.fixture()
    def docs(self) -> list[Document]:
        print("Generating docs")
        return [
            Document(
                text_representation=f"Document {i}",
                doc_id=i,
                embedding=[1.1, 2.2, 3.3, 4.4, 5.5],
                properties={"document_number": i},
            )
            for i in range(100)
        ]

    @pytest.fixture()
    def docset(self, docs: list[Document]) -> DocSet:
        context = sycamore.init()
        return context.read.document(docs)

    def test_kmeans(self, docset: DocSet):
        centroids = docset.kmeans(3, 4)
        assert len(centroids) == 3

    def test_closest(self):
        row = [[0, 0, 0, 0]]
        centroids = [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [-1, -1, -1, -1],
        ]
        assert KMeans.closest(row, centroids) == 0

    def test_converged(self):
        last_ones = [[1.0, 1.0], [10.0, 10.0]]
        next_ones = [[2.0, 2.0], [12.0, 12.0]]
        assert KMeans.converged(last_ones, next_ones, 10) == True
        assert KMeans.converged(last_ones, next_ones, 1) == False

    def test_converge(self):
        import numpy as np

        points = np.random.uniform(0, 10, (20, 4))
        embeddings = [{"vector": list(point), "cluster": -1} for point in points]
        embeddings = ray.data.from_items(embeddings)
        centroids = [[2.0, 2.0, 2.0, 2.0], [8.0, 8.0, 8.0, 8.0]]
        new_centroids = KMeans.update(embeddings, centroids, 2, 1e-4)
        assert len(new_centroids) == 2
