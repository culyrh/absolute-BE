"""
협업 필터링 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class CollaborativeAlgorithm(BaseRecommendationAlgorithm):
    """협업 필터링 기반 추천"""
    
    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str], recommend_result: pd.DataFrame):
        super().__init__(centroids, norm_cols)
        self.recommend_result = recommend_result
    
    @property
    def name(self) -> str:
        return "collaborative"
    
    @property
    def description(self) -> str:
        return "협업 필터링 방식으로 유사한 주유소의 용도를 기반으로 추천합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """협업 필터링 기반 추천"""
        try:
            if len(df) == 0:
                return []
            
            address_row = df.iloc[0]
            available_cols = [col for col in self.norm_cols if col in address_row.index]
            
            if not available_cols:
                return []
            
            address_vector = np.array([address_row[col] for col in available_cols]).reshape(1, -1)
            
            # 추천 결과 데이터에서 유사한 주유소 찾기
            if self.recommend_result is None or len(self.recommend_result) == 0:
                return []
            
            # 사용 가능한 컬럼만 필터링
            result_available_cols = [col for col in available_cols if col in self.recommend_result.columns]
            
            if not result_available_cols:
                return []
            
            # 각 주유소와의 유사도 계산
            similarities = []
            
            for idx, row in self.recommend_result.iterrows():
                # 특징 벡터 추출
                row_vector = np.array([row.get(col, 0) for col in result_available_cols]).reshape(1, -1)
                
                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    address_vector[:, :len(result_available_cols)], 
                    row_vector
                )[0][0]
                
                usage_type = row.get("대분류", "")
                
                if usage_type:
                    similarities.append({
                        "usage_type": usage_type,
                        "similarity": float(similarity)
                    })
            
            # 용도별로 그룹화하여 평균 유사도 계산
            if not similarities:
                return []
            
            df_sim = pd.DataFrame(similarities)
            grouped = df_sim.groupby("usage_type")["similarity"].mean().reset_index()
            grouped = grouped.sort_values("similarity", ascending=False).head(top_k)
            
            recommendations = []
            for i, (_, row) in enumerate(grouped.iterrows()):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=row["usage_type"],
                        score=float(row["similarity"]),
                        rank=i + 1,
                        avg_similarity=float(row["similarity"])
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 협업 필터링 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []