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
        return "유클리드 거리를 사용하여 특징 벡터 간 거리를 계산합니다. 거리가 가까울수록 유사합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        유클리드 거리 기반 추천
        
        주유소의 특징 벡터와 각 용도 유형의 센트로이드 벡터 간
        유클리드 거리를 계산하여 가장 가까운 용도를 추천합니다.
        """
        try:
            if len(df) == 0 or self.centroids is None or len(self.centroids) == 0:
                return []
            
            # 첫 번째 주소 사용
            address_row = df.iloc[0]
            
            # 사용 가능한 특징 컬럼 확인
            available_cols = [col for col in self.norm_cols if col in address_row.index and col in self.centroids.columns]
            
            if not available_cols:
                return []
            
            # 주소의 특징 벡터
            address_vector = np.array([address_row[col] for col in available_cols])
            
            # 각 용도 유형별 거리 계산
            distances = []
            
            for _, centroid_row in self.centroids.iterrows():
                usage_type = centroid_row.get("usage_type", "")
                
                # 센트로이드 벡터
                centroid_vector = np.array([centroid_row[col] for col in available_cols])
                
                # 유클리드 거리 계산
                distance = euclidean(address_vector, centroid_vector)
                
                distances.append({
                    "usage_type": usage_type,
                    "distance": float(distance)
                })
            
            # 거리 기준 정렬 (오름차순 - 거리가 가까울수록 좋음)
            distances.sort(key=lambda x: x["distance"])
            
            # 상위 top_k개 선택
            top_distances = distances[:top_k]
            
            # 거리를 유사도 점수로 변환 (0~1 범위)
            # 점수 = 1 / (1 + 정규화된_거리)
            max_distance = max([d["distance"] for d in top_distances]) if top_distances else 1
            
            # 결과 포맷팅
            recommendations = []
            for i, dist in enumerate(top_distances):
                # 거리를 유사도 점수로 변환
                normalized_distance = dist["distance"] / max_distance if max_distance > 0 else 0
                similarity_score = 1 / (1 + normalized_distance)
                
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=dist["usage_type"],
                        score=similarity_score,
                        rank=i + 1,
                        distance=dist["distance"],
                        similarity=similarity_score
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 유클리드 거리 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []