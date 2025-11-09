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
        """피어슨 상관계수 기반 추천"""
        try:
            if len(df) == 0 or self.centroids is None or len(self.centroids) == 0:
                return []
            
            address_row = df.iloc[0]
            available_cols = [col for col in self.norm_cols if col in address_row.index and col in self.centroids.columns]
            
            if not available_cols or len(available_cols) < 2:
                return []
            
            address_vector = np.array([address_row[col] for col in available_cols])
            
            correlations = []
            for _, centroid_row in self.centroids.iterrows():
                usage_type = centroid_row.get("usage_type", "")
                centroid_vector = np.array([centroid_row[col] for col in available_cols])
                
                try:
                    correlation, _ = pearsonr(address_vector, centroid_vector)
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    correlations.append({
                        "usage_type": usage_type,
                        "correlation": float(correlation)
                    })
                except:
                    correlations.append({
                        "usage_type": usage_type,
                        "correlation": 0.0
                    })
            
            correlations.sort(key=lambda x: x["correlation"], reverse=True)
            top_correlations = correlations[:top_k]
            
            recommendations = []
            for i, corr in enumerate(top_correlations):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=corr["usage_type"],
                        score=corr["correlation"],
                        rank=i + 1,
                        correlation=corr["correlation"]
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 피어슨 상관계수 추천 실패: {str(e)}")
            return []