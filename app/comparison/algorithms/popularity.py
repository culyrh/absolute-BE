"""
인기도 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class PopularityAlgorithm(BaseRecommendationAlgorithm):
    """인기도 기반 추천"""
    
    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str], recommend_result: pd.DataFrame):
        super().__init__(centroids, norm_cols)
        self.recommend_result = recommend_result
    
    @property
    def name(self) -> str:
        return "popularity"
    
    @property
    def description(self) -> str:
        return "용도 유형의 빈도수를 기반으로 가장 인기 있는 용도를 추천합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """인기도 기반 추천"""
        try:
            if len(df) == 0:
                return []
            
            address_row = df.iloc[0]
            
            if "대분류" not in self.recommend_result.columns:
                return []
            
            usage_type_counts = self.recommend_result["대분류"].value_counts().reset_index()
            usage_type_counts.columns = ["usage_type", "count"]
            
            top_usage_types = usage_type_counts.head(top_k)
            
            recommendations = []
            max_count = usage_type_counts["count"].max()
            
            for i, (_, row) in enumerate(top_usage_types.iterrows()):
                usage_type = row["usage_type"]
                count = row["count"]
                
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=usage_type,
                        score=float(count / max_count),
                        rank=i + 1,
                        popularity_count=int(count),
                        popularity_ratio=float(count / usage_type_counts["count"].sum())
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 인기도 기반 추천 실패: {str(e)}")
            return []