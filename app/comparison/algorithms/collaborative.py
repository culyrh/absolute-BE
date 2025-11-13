import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseAlgorithm


class CollaborativeAlgorithm(BaseAlgorithm):
    """협업 필터링 기반 추천 (센트로이드 유사도 기반)"""

    def recommend(self, test_df, top_k=5):
        test_df = self._normalize_if_missing(test_df)
        results = []

        for _, row in test_df.iterrows():
            region = row.get("관할주소") or row.get("권역")
            region_centroids = self._filter_by_region(self.centroids, region)

            input_vec = np.array([[row[col] for col in self.norm_cols if col in row]])
            centroid_vecs = region_centroids[self.norm_cols].fillna(0).to_numpy()

            sims = cosine_similarity(input_vec, centroid_vecs)[0]
            region_centroids = region_centroids.copy()
            region_centroids["similarity"] = sims

            top = region_centroids.sort_values("similarity", ascending=False).head(top_k)
            for _, c_row in top.iterrows():
                results.append({
                    "usage_type": self._extract_usage_name(c_row),
                    "score": c_row["similarity"]
                })

        return results
