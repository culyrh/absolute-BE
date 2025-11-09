"""
코사인 유사도 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class CosineSimilarityAlgorithm(BaseRecommendationAlgorithm):
    """코사인 유사도 기반 추천"""
    
    @property
    def name(self) -> str:
        return "cosine_similarity"
    
    @property
    def description(self) -> str:
        return "코사인 유사도를 사용하여 특징 벡터 간 유사도를 계산합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        코사인 유사도 기반 추천
        
        주유소의 특징 벡터와 각 용도 유형의 센트로이드 벡터 간
        코사인 유사도를 계산하여 가장 유사한 용도를 추천합니다.
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
            address_vector = np.array([address_row[col] for col in available_cols]).reshape(1, -1)
            
            # 각 용도 유형별 유사도 계산
            similarities = []
            
            for _, centroid_row in self.centroids.iterrows():
                usage_type = centroid_row.get("usage_type", "")
                
                # 센트로이드 벡터
                centroid_vector = np.array([centroid_row[col] for col in available_cols]).reshape(1, -1)
                
                # 코사인 유사도 계산
                similarity = cosine_similarity(address_vector, centroid_vector)[0][0]
                
                similarities.append({
                    "usage_type": usage_type,
                    "similarity": float(similarity)
                })
            
            # 유사도 기준 정렬
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 상위 top_k개 선택
            top_similarities = similarities[:top_k]
            
            # 결과 포맷팅
            recommendations = []
            for i, sim in enumerate(top_similarities):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=sim["usage_type"],
                        score=sim["similarity"],
                        rank=i + 1,
                        similarity=sim["similarity"]
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 코사인 유사도 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []