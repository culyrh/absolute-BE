from .base import BaseAlgorithm
import numpy as np


class PopularityAlgorithm(BaseAlgorithm):
    """인기도 기반 추천 (단순 빈도 기반 예시)"""

    def recommend(self, test_df, top_k=5):
        if self.train_data is None:
            return []

        freq = self.train_data["추천_대분류"].value_counts(normalize=True)
        results = [{"usage_type": k, "score": v} for k, v in freq.head(top_k).items()]
        return results
