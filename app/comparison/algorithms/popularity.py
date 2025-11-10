"""
인기도 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class PopularityAlgorithm(BaseRecommendationAlgorithm):
    """인기도 기반 추천"""
    
    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str], train_data: pd.DataFrame):
        """
        Args:
            centroids: 센트로이드 데이터프레임
            norm_cols: 정규화된 특징 컬럼 리스트
            train_data: 학습 데이터 (추천결과_행단위.csv)
        """
        super().__init__(centroids, norm_cols)
        self.train_data = train_data
    
    @property
    def name(self) -> str:
        return "popularity"
    
    @property
    def description(self) -> str:
        return "학습 데이터에서 가장 많이 등장한 용도 유형을 우선 추천합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        인기도 기반 추천
        
        학습 데이터에서 대분류의 빈도수를 계산하여
        가장 많이 등장한 용도 유형을 우선 추천합니다.
        """
        try:
            if len(df) == 0:
                return []
            
            # 첫 번째 주소 사용
            address_row = df.iloc[0]
            
            # 대분류 빈도수 계산
            if "대분류" not in self.train_data.columns:
                return []
            
            usage_type_counts = self.train_data["대분류"].value_counts().reset_index()
            usage_type_counts.columns = ["usage_type", "count"]
            
            # 상위 top_k개 선택
            top_usage_types = usage_type_counts.head(top_k)
            
            # 정규화된 점수 계산 (최대값 기준)
            max_count = top_usage_types["count"].max() if len(top_usage_types) > 0 else 1
            
            # 결과 포맷팅
            recommendations = []
            
            for i, (_, row) in enumerate(top_usage_types.iterrows()):
                usage_type = row["usage_type"]
                count = row["count"]
                score = float(count / max_count)
                
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=usage_type,
                        score=score,
                        rank=i + 1,
                        popularity_count=int(count),
                        popularity_score=score
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 인기도 기반 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []