import numpy as np
from scipy.stats import pearsonr
from .base import BaseAlgorithm


class PearsonCorrelationAlgorithm(BaseAlgorithm):
    """피어슨 상관계수 기반 추천"""

    def recommend(self, test_df, top_k=5):
        test_df = self._normalize_if_missing(test_df)
        results = []

        for _, row in test_df.iterrows():
            region = row.get("관할주소") or row.get("권역")
            region_centroids = self._filter_by_region(self.centroids, region)

            input_vec = np.array([row[col] for col in self.norm_cols if col in row])

            scores = []
            for _, c_row in region_centroids.iterrows():
                c_vec = np.array([c_row[col] for col in self.norm_cols if col in c_row])
                corr, _ = pearsonr(input_vec, c_vec) if len(input_vec) == len(c_vec) else (0, None)
                scores.append(corr)

            region_centroids = region_centroids.copy()
            region_centroids["similarity"] = scores
            top = region_centroids.sort_values("similarity", ascending=False).head(top_k)

            for _, c_row in top.iterrows():
                results.append({
                    "usage_type": self._extract_usage_name(c_row),
                    "score": c_row["similarity"]
                })

        return results
