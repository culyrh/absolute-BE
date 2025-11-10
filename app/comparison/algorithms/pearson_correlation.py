"""
피어슨 상관계수 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class PearsonCorrelationAlgorithm(BaseRecommendationAlgorithm):
    """피어슨 상관계수 기반 추천"""
    
    @property
    def name(self) -> str:
        return "pearson_correlation"
    
    @property
    def description(self) -> str:
        return "피어슨 상관계수를 사용하여 특징 벡터 간 선형 상관관계를 계산합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        피어슨 상관계수 기반 추천
        
        주유소의 특징 벡터와 각 용도 유형의 센트로이드 벡터 간
        피어슨 상관계수를 계산하여 가장 상관관계가 높은 용도를 추천합니다.
        """
        try:
            if len(df) == 0 or self.centroids is None or len(self.centroids) == 0:
                return []
            
            # 첫 번째 주소 사용
            address_row = df.iloc[0]
            
            # 사용 가능한 특징 컬럼 확인
            available_cols = [col for col in self.norm_cols if col in address_row.index and col in self.centroids.columns]
            
            if not available_cols or len(available_cols) < 2:
                # 피어슨 상관계수는 최소 2개 이상의 변수가 필요
                return []
            
            # 주소의 특징 벡터
            address_vector = np.array([address_row[col] for col in available_cols])
            
            # 각 용도 유형별 상관계수 계산
            correlations = []
            
            for _, centroid_row in self.centroids.iterrows():
                usage_type = centroid_row.get("usage_type", "")
                
                # 센트로이드 벡터
                centroid_vector = np.array([centroid_row[col] for col in available_cols])
                
                # 피어슨 상관계수 계산
                try:
                    # 벡터의 분산이 0인 경우 처리
                    if np.std(address_vector) == 0 or np.std(centroid_vector) == 0:
                        correlation = 0.0
                    else:
                        correlation, _ = pearsonr(address_vector, centroid_vector)
                        
                        # NaN 처리
                        if np.isnan(correlation):
                            correlation = 0.0
                except Exception:
                    correlation = 0.0
                
                correlations.append({
                    "usage_type": usage_type,
                    "correlation": float(correlation)
                })
            
            # 상관계수 기준 정렬 (내림차순 - 상관계수가 높을수록 좋음)
            correlations.sort(key=lambda x: x["correlation"], reverse=True)
            
            # 상위 top_k개 선택
            top_correlations = correlations[:top_k]
            
            # 결과 포맷팅
            recommendations = []
            for i, corr in enumerate(top_correlations):
                # 상관계수를 0~1 범위로 정규화 (-1~1 -> 0~1)
                normalized_score = (corr["correlation"] + 1) / 2
                
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=corr["usage_type"],
                        score=normalized_score,
                        rank=i + 1,
                        correlation=corr["correlation"],
                        normalized_score=normalized_score
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 피어슨 상관계수 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []