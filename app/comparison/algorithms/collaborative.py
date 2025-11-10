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
        return "collaborative_filtering"
    
    @property
    def description(self) -> str:
        return "협업 필터링을 사용하여 유사한 지역의 용도 유형 패턴을 기반으로 추천합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        협업 필터링 기반 추천
        
        입력 주소와 유사한 특징을 가진 지역들의 용도 유형 패턴을 분석하여
        가장 적합한 용도를 추천합니다.
        """
        try:
            if len(df) == 0 or len(self.train_data) == 0:
                return []
            
            # 첫 번째 주소 사용
            address_row = df.iloc[0]
            
            # 사용 가능한 특징 컬럼 확인
            available_cols = [col for col in self.norm_cols if col in address_row.index and col in self.train_data.columns]
            
            if not available_cols:
                return []
            
            # 주소의 특징 벡터
            address_vector = np.array([address_row[col] for col in available_cols]).reshape(1, -1)
            
            # 학습 데이터의 특징 벡터들과 유사도 계산
            train_vectors = self.train_data[available_cols].values
            similarities = cosine_similarity(address_vector, train_vectors)[0]
            
            # 유사도를 학습 데이터에 추가
            train_with_sim = self.train_data.copy()
            train_with_sim["similarity"] = similarities
            
            # 유사도가 높은 상위 K개 선택 (K = top_k * 3)
            k_neighbors = min(len(train_with_sim), top_k * 3)
            top_similar = train_with_sim.nlargest(k_neighbors, "similarity")
            
            # 대분류가 없으면 빈 결과 반환
            if "대분류" not in top_similar.columns:
                return []
            
            # 유사한 지역들의 대분류 빈도수 계산 (유사도 가중치 적용)
            usage_scores = {}
            
            for _, row in top_similar.iterrows():
                usage_type = row["대분류"]
                similarity = row["similarity"]
                
                if usage_type not in usage_scores:
                    usage_scores[usage_type] = 0
                
                # 유사도를 가중치로 사용
                usage_scores[usage_type] += similarity
            
            # 점수 기준 정렬
            sorted_scores = sorted(
                [(usage_type, score) for usage_type, score in usage_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # 상위 top_k개 선택
            top_scores = sorted_scores[:top_k]
            
            # 정규화된 점수 계산
            max_score = top_scores[0][1] if top_scores else 1
            
            # 결과 포맷팅
            recommendations = []
            
            for i, (usage_type, score) in enumerate(top_scores):
                normalized_score = float(score / max_score) if max_score > 0 else 0
                
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=usage_type,
                        score=normalized_score,
                        rank=i + 1,
                        cf_score=float(score),
                        normalized_score=normalized_score,
                        neighbor_count=k_neighbors
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ 협업 필터링 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []