import numpy as np
from .base import BaseAlgorithm


class EuclideanDistanceAlgorithm(BaseAlgorithm):
    """유클리드 거리 기반 추천"""

    def recommend(self, test_df, top_k=5):
        test_df = self._normalize_if_missing(test_df)
        results = []

        for _, row in test_df.iterrows():
            region = row.get("관할주소") or row.get("권역")
            region_centroids = self._filter_by_region(self.centroids, region)

            input_vec = np.array([row[col] for col in self.norm_cols if col in row])
            centroid_vecs = region_centroids[self.norm_cols].fillna(0).to_numpy()

            dists = np.linalg.norm(centroid_vecs - input_vec, axis=1)
            region_centroids = region_centroids.copy()
            region_centroids["distance"] = dists

            top = region_centroids.sort_values("distance", ascending=True).head(top_k)
            for _, c_row in top.iterrows():
                results.append({
                    "usage_type": self._extract_usage_name(c_row),
                    "score": 1 / (1 + c_row["distance"])
                })

        return results
