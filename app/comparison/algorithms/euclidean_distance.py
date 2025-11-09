"""
유클리드 거리 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class EuclideanDistanceAlgorithm(BaseRecommendationAlgorithm):
    """유클리드 거리 기반 추천"""
    
    @property
    def name(self) -> str:
        return "euclidean_distance"
    
    @property
    def description(self) -> str:
        return "유클리드 거리를 사용하여 특징 벡터 간 거리를 계산합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """유클리드 거리 기반 추천"""
        try:
            if len(df) == 0 or self.centroids is None or len(self.centroids) == 0:
                return []
            
            address_row = df.iloc[0]
            available_cols = [col for col in self.norm_cols if col in address_row.index and col in self.centroids.columns]
            
            if not available_cols:
                return []
            
            address_vector = np.array([address_row[col] for col in available_cols])
            
            distances = []
            for _, centroid_row in self.centroids.iterrows():
                usage_type = centroid_row.get("usage_type", "")
                centroid_vector = np.array([centroid_row[col] for col in available_cols])
                
                # 유클리드 거리 계산
                distance = euclidean(address_vector, centroid_vector)
                
                # 거리를 유사도로 변환 (거리가 작을수록 유사도 높음)
                similarity = 1 / (1 + distance)
                
                distances.append({
                    "usage_type": usage_type,
                    "distance": float(distance),
                    "similarity": float(similarity)
                })
            
            # 유사도 기준 정렬 (거리가 가까울수록 좋음)
            distances.sort(key=lambda x: x["similarity"], reverse=True)
            
            top_distances = distances[:top_k]
            
            recommendations = []
            for i, dist in enumerate(top_distances):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=dist["usage_type"],
                        score=dist["similarity"],
                        rank=i + 1,
                        distance=dist["distance"],
                        similarity=dist["similarity"]
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 유클리드 거리 추천 실패: {str(e)}")
            return []