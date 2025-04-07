import numpy as np


class KMeans:

    @staticmethod
    def closest(row, centroids):
        import torch

        row = torch.Tensor([row])
        centroids = torch.Tensor(centroids)
        distance = torch.cdist(row, centroids)
        idx = torch.argmin(distance)
        return idx

    @staticmethod
    def converged(last_ones, next_ones, epsilon):
        import torch

        distance = torch.cdist(torch.Tensor(last_ones), torch.Tensor(next_ones))
        return len(last_ones) == torch.sum(distance < epsilon)

    @staticmethod
    def random_init(embeddings, K):
        count = embeddings.count()
        assert count > 0
        K = K if count > K else count
        fraction = min(2 * K / count, 1.0)

        candidates = np.array([c["vector"] for c in embeddings.random_sample(fraction).take()])
        print(candidates)

        centroids = np.unique(candidates, axis=0)[:K].tolist()
        return centroids

    @staticmethod
    def init(embeddings, K, init_mode):
        if init_mode == "random":
            return KMeans.random_init(embeddings, K)
        else:
            raise Exception("Unknown init mode")

    @staticmethod
    def update(embeddings, centroids, iterations, epsilon):
        i = 0
        d = len(centroids[0])

        from ray.data.aggregate import AggregateFn

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
