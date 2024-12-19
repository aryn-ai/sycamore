import torch
from ray.data.aggregate import AggregateFn


class KMeans:

    @staticmethod
    def closest(row, centroids):
        row = torch.Tensor([row])
        centroids = torch.Tensor(centroids)
        distance = torch.cdist(row, centroids)
        id = torch.argmin(distance)
        return id

    @staticmethod
    def converged(last_ones, next_ones, epsilon):
        # TODO, need accumulate the cost also
        distance = torch.cdist(torch.Tensor(last_ones), torch.Tensor(next_ones))
        return len(last_ones) == torch.sum(distance < epsilon)

    @staticmethod
    def init(embeddings, K):
        # TODO,
        #  1. fix this random, guarantee K different samples
        #  2. take the k-means|| as initialization
        sampled = embeddings.take(K)
        centroids = [s["vector"] for s in sampled]
        return centroids

    @staticmethod
    def update(embeddings, centroids, iterations, epsilon):
        i = 0
        d = len(centroids[0])

        update_centroids = AggregateFn(
            init=lambda v: ([0] * d, 0),
            accumulate_row=lambda a, row: ([x + y for x, y in zip(a[0], row["vector"])], a[1] + 1),
            merge=lambda a1, a2: ([x + y for x, y in zip(a1[0], a2[0])], a1[1] + a2[1]),
            name="centroids",
        )

        while i < iterations:

            def _find_cluster(row):
                idx = KMeans.closest(row["vector"], centroids)
                return {"vector": row["vector"], "cluster": idx}

            aggregated = embeddings.map(_find_cluster).groupby("cluster").aggregate(update_centroids).take()
            import numpy as np

            new_centroids = [list(np.array(c["centroids"][0]) / c["centroids"][1]) for c in aggregated]

            if KMeans.converged(centroids, new_centroids, epsilon):
                return new_centroids
            else:
                i += 1
                centroids = new_centroids

        return centroids
