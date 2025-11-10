"""
AHP-TOPSIS 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from app.comparison.algorithms.base import BaseRecommendationAlgorithm


class AHPTopsisAlgorithm(BaseRecommendationAlgorithm):
    """AHP-TOPSIS 기반 추천"""
    
    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str], train_data: pd.DataFrame):
        """
        Args:
            centroids: 센트로이드 데이터프레임
            norm_cols: 정규화된 특징 컬럼 리스트
            train_data: 학습 데이터 (추천결과_행단위.csv)
        """
        super().__init__(centroids, norm_cols)
        self.train_data = train_data
        
        # AHP 가중치 정의 (고정값)
        self.weights = {
            "인구[명]_norm": 0.30,
            "교통량_norm": 0.25,
            "숙박업소(관광지수)_norm": 0.20,
            "상권밀집도(비율)_norm": 0.25
        }
    
    @property
    def name(self) -> str:
        return "ahp_topsis"
    
    @property
    def description(self) -> str:
        return "AHP로 계산된 가중치와 TOPSIS 다기준 의사결정을 사용하여 최적 용도를 추천합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        AHP-TOPSIS 기반 추천
        
        1. AHP로 각 평가 지표의 가중치 계산
        2. TOPSIS로 다기준 의사결정 수행
        3. 최적의 용도 유형 추천
        """
        try:
            if len(df) == 0 or len(self.train_data) == 0:
                return []
            
            # 첫 번째 주소 사용
            address_row = df.iloc[0]
            
            # 권역 추출
            region = address_row.get("권역", "")
            if not region and "주소" in address_row:
                # 주소에서 권역 추출 시도
                address = str(address_row["주소"])
                for r in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", 
                         "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]:
                    if r in address:
                        region = r
                        break
            
            # 사용 가능한 특징 컬럼 확인
            available_cols = [col for col in self.weights.keys() if col in address_row.index]
            
            if not available_cols:
                return []
            
            # 권역별 데이터 필터링
            region_df = self.train_data
            if region and "권역" in self.train_data.columns:
                region_df = self.train_data[self.train_data["권역"].astype(str).str.contains(region, na=False)]
            
            # 대분류가 없으면 빈 결과 반환
            if "대분류" not in region_df.columns:
                return []
            
            usage_types = region_df["대분류"].unique()
            
            if len(usage_types) == 0:
                return []
            
            # 1. 각 용도 유형별 중위값 계산 (결정 행렬)
            decision_matrix = {}
            
            for usage_type in usage_types:
                type_df = region_df[region_df["대분류"] == usage_type]
                
                if len(type_df) > 0:
                    medians = {}
                    for col in available_cols:
                        if col in type_df.columns:
                            medians[col] = type_df[col].median()
                        else:
                            medians[col] = 0.5
                    decision_matrix[usage_type] = medians
                else:
                    decision_matrix[usage_type] = {col: 0.5 for col in available_cols}
            
            # 2. 대상 주유소의 지표값
            site_values = {col: float(address_row.get(col, 0)) for col in available_cols}
            
            # 3. 유사도 점수 행렬 생성
            similarity_matrix = {}
            
            for usage_type, medians in decision_matrix.items():
                distances = {}
                for col in available_cols:
                    # 절대 거리 계산 후 유사도로 변환
                    distance = abs(site_values[col] - medians[col])
                    distances[col] = 1 - distance
                similarity_matrix[usage_type] = distances
            
            # 4. 가중치 적용
            weighted_matrix = {}
            
            for usage_type, similarities in similarity_matrix.items():
                weighted = {}
                for col in available_cols:
                    weighted[col] = self.weights.get(col, 0) * similarities[col]
                weighted_matrix[usage_type] = weighted
            
            # 5. 이상해(Ideal Positive)와 반대해(Ideal Negative) 계산
            ideal_positive = {}
            ideal_negative = {}
            
            for col in available_cols:
                values = [weighted_matrix[ut][col] for ut in weighted_matrix]
                ideal_positive[col] = max(values)
                ideal_negative[col] = min(values)
            
            # 6. 각 대안과 이상해/반대해 간의 거리 계산
            distances_positive = {}
            distances_negative = {}
            
            for usage_type, weighted in weighted_matrix.items():
                # 이상해와의 거리
                dist_pos = sum((weighted[col] - ideal_positive[col]) ** 2 for col in available_cols) ** 0.5
                # 반대해와의 거리
                dist_neg = sum((weighted[col] - ideal_negative[col]) ** 2 for col in available_cols) ** 0.5
                
                distances_positive[usage_type] = dist_pos
                distances_negative[usage_type] = dist_neg
            
            # 7. 상대 근접도 계산 (TOPSIS 점수)
            closeness = {}
            
            for usage_type in weighted_matrix:
                d_pos = distances_positive[usage_type]
                d_neg = distances_negative[usage_type]
                
                # 0으로 나누기 방지
                if d_pos + d_neg == 0:
                    closeness[usage_type] = 0
                else:
                    closeness[usage_type] = d_neg / (d_pos + d_neg)
            
            # 8. 결과 정렬 및 상위 추천
            sorted_results = sorted(
                [(usage_type, score) for usage_type, score in closeness.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # 상위 top_k개 선택
            top_results = sorted_results[:top_k]
            
            # 9. 결과 포맷팅
            recommendations = []
            
            for i, (usage_type, score) in enumerate(top_results):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=usage_type,
                        score=float(score),
                        rank=i + 1,
                        topsis_score=float(score),
                        ahp_weights=self.weights,
                        region=region
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ AHP-TOPSIS 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []