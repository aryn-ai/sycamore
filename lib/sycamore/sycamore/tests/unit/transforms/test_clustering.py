import numpy as np
import pytest
import ray.data

import sycamore
from sycamore.data import Document
from sycamore.transforms.clustering import KMeans


class TestKMeans:

    @pytest.mark.skip(reason="flaky")
    def test_kmeans(self):
        np.random.seed(2024)
        points = np.random.uniform(0, 40, (20, 4))
        docs = [
            Document(text_representation=f"Document {i}", doc_id=i, embedding=point, properties={"document_number": i})
            for i, point in enumerate(points)
        ]
        context = sycamore.init()
        docset = context.read.document(docs)
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

    def test_random(self):
        np.random.seed(2024)
        points = np.random.uniform(0, 40, (20, 4))
        embeddings = [{"vector": list(point), "cluster": -1} for point in points]
        embeddings = ray.data.from_items(embeddings)
        centroids = KMeans.random_init(embeddings, 10)
        assert len(centroids) == 10

    def test_converged(self):
        last_ones = [[1.0, 1.0], [10.0, 10.0]]
        next_ones = [[2.0, 2.0], [12.0, 12.0]]
        assert KMeans.converged(last_ones, next_ones, 10).item() is True
        assert KMeans.converged(last_ones, next_ones, 1).item() is False

    def test_converge(self):
        np.random.seed(2024)
        points = np.random.uniform(0, 10, (20, 4))
        embeddings = [{"vector": list(point), "cluster": -1} for point in points]
        embeddings = ray.data.from_items(embeddings)
        centroids = [[2.0, 2.0, 2.0, 2.0], [8.0, 8.0, 8.0, 8.0]]
        new_centroids = KMeans.update(embeddings, centroids, 2, 1e-4)
        assert len(new_centroids) == 2

    def test_clustering(self):
        np.random.seed(2024)
        points = np.random.uniform(0, 40, (20, 4))
        docs = [
            Document(text_representation=f"Document {i}", doc_id=i, embedding=point, properties={"document_number": i})
            for i, point in enumerate(points)
        ]
        context = sycamore.init()
        docset = context.read.document(docs)
        centroids = docset.kmeans(3, 4)

        clustered_docs = docset.clustering(centroids, "cluster").take_all()
        ids = [doc["cluster"] for doc in clustered_docs]
        assert all(0 <= idx < 3 for idx in ids)
