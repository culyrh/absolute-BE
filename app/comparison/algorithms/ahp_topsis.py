"""
AHP-TOPSIS 기반 추천 알고리즘
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from app.comparison.algorithms.base import BaseRecommendationAlgorithm
from app.utils.preprocessing import extract_province, normalize_region


class AHPTopsisAlgorithm(BaseRecommendationAlgorithm):
    """AHP-TOPSIS 다기준 의사결정 기반 추천"""
    
    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str], recommend_result: pd.DataFrame):
        super().__init__(centroids, norm_cols)
        self.recommend_result = recommend_result
    
    @property
    def name(self) -> str:
        return "ahp_topsis"
    
    @property
    def description(self) -> str:
        return "AHP와 TOPSIS를 결합한 다기준 의사결정 방법으로 추천합니다."
    
    def recommend(self, df: pd.DataFrame, top_k: int = 10, region: str = None) -> List[Dict[str, Any]]:
        """AHP-TOPSIS 기반 추천"""
        try:
            if len(df) == 0:
                return []
            
            address_row = df.iloc[0]
            
            # 권역 추출
            address_region = region
            if not address_region:
                address = address_row.get("주소", "")
                address_region = extract_province(address)
                
                if address_region:
                    address_region = normalize_region(address_region)
                else:
                    address_region = "전라북도"
            
            # 추천 결과 데이터 가져오기
            if self.recommend_result is None or len(self.recommend_result) == 0:
                return []
            
            # 권역별 데이터 필터링
            region_df = self.recommend_result[self.recommend_result["권역"] == address_region] if "권역" in self.recommend_result.columns else self.recommend_result
            
            if len(region_df) == 0:
                return []
            
            # AHP 가중치 정의
            weights = {
                "인구[명]_norm": 0.30,
                "교통량_norm": 0.25,
                "숙박업소(관광지수)_norm": 0.20,
                "상권밀집도(비율)_norm": 0.25
            }
            
            available_cols = [col for col in weights.keys() if col in address_row.index]
            
            if not available_cols:
                return []
            
            # 용도 유형 추출
            usage_types = region_df["대분류"].unique() if "대분류" in region_df.columns else []
            
            if len(usage_types) == 0:
                usage_types = [
                    "근린생활시설", "공동주택", "자동차관련시설", 
                    "판매시설", "업무시설", "숙박시설",
                    "공장", "가설건축", "기타"
                ]
            
            # 결정 행렬 생성
            decision_matrix = {}
            
            for usage_type in usage_types:
                type_df = region_df[region_df["대분류"] == usage_type] if "대분류" in region_df.columns else pd.DataFrame()
                
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
            
            # 대상 주유소의 지표 값 추출
            site_values = {}
            for col in available_cols:
                site_values[col] = float(address_row.get(col, 0))
            
            # TOPSIS 알고리즘 적용
            similarity_matrix = {}
            
            for usage_type, medians in decision_matrix.items():
                distances = {}
                
                for col in available_cols:
                    distance = abs(site_values[col] - medians[col])
                    distances[col] = 1 - distance
                
                similarity_matrix[usage_type] = distances
            
            # 정규화 및 가중치 적용
            weighted_matrix = {}
            
            for usage_type, similarities in similarity_matrix.items():
                weighted = {}
                
                for col in available_cols:
                    weighted[col] = weights.get(col, 0) * similarities[col]
                
                weighted_matrix[usage_type] = weighted
            
            # 이상해 및 반대해 계산
            ideal_positive = {}
            ideal_negative = {}
            
            for col in available_cols:
                max_val = max(weighted_matrix[ut][col] for ut in weighted_matrix)
                min_val = min(weighted_matrix[ut][col] for ut in weighted_matrix)
                
                ideal_positive[col] = max_val
                ideal_negative[col] = min_val
            
            # 거리 계산
            distances_positive = {}
            distances_negative = {}
            
            for usage_type, weighted in weighted_matrix.items():
                dist_pos = sum((weighted[col] - ideal_positive[col]) ** 2 for col in available_cols) ** 0.5
                dist_neg = sum((weighted[col] - ideal_negative[col]) ** 2 for col in available_cols) ** 0.5
                
                distances_positive[usage_type] = dist_pos
                distances_negative[usage_type] = dist_neg
            
            # 상대 근접도 계산
            closeness = {}
            
            for usage_type in weighted_matrix:
                d_pos = distances_positive[usage_type]
                d_neg = distances_negative[usage_type]
                
                if d_pos + d_neg == 0:
                    closeness[usage_type] = 0
                else:
                    closeness[usage_type] = d_neg / (d_pos + d_neg)
            
            # 결과 정렬 및 상위 추천 반환
            sorted_results = sorted(
                [(usage_type, score) for usage_type, score in closeness.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            top_results = sorted_results[:top_k]
            
            recommendations = []
            
            for i, (usage_type, score) in enumerate(top_results):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=usage_type,
                        score=float(score),
                        rank=i + 1,
                        ahp_weights=weights,
                        region=address_region
                    )
                )
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ AHP-TOPSIS 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []