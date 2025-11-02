"""
추천 시스템 서비스
"""

import pandas as pd
import numpy as np
import re
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

from app.utils.data_loader import load_all_data, find_column_by_keyword
from app.utils.preprocessing import (
    preprocess_gas_station_data, merge_with_stats, normalize_features,
    categorize_by_usage_type_and_region, calculate_centroids,
    extract_admin_region, extract_province, normalize_region
)
from app.schemas.recommendation import RecommendationAlgorithm, RecommendationResponse
from app.core.config import settings


class RecommendationService:
    """추천 시스템 서비스"""
    
    def __init__(self):
        self.data = None
        self.centroids = None
        self.feature_cols = ["인구[명]", "교통량", "숙박업소(관광지수)", "상권밀집도(비율)", "공시지가(토지단가)"]
        self.norm_cols = [f"{col}_norm" for col in self.feature_cols]
        self.initialize_data()
    
    def initialize_data(self):
        """데이터 초기화 및 로드"""
        print("🚀 추천 서비스 초기화 중...")
        
        # 모든 데이터 로드
        self.data = load_all_data()
        
        # 주유소 데이터 전처리
        self.data["gas_station"] = preprocess_gas_station_data(self.data["gas_station"])
        
        # 인구수와 사업체 데이터 병합
        self.data["gas_station"] = merge_with_stats(
            self.data["gas_station"],
            self.data["population"],
            self.data["business"]
        )
        
        # 특징 정규화
        available_cols = [col for col in self.feature_cols if col in self.data["gas_station"].columns]
        self.data["gas_station"] = normalize_features(self.data["gas_station"], available_cols)
        
        # 센트로이드 데이터 처리
        self.process_centroids()
        
        print("✅ 추천 서비스 초기화 완료")
    
    def process_centroids(self):
        """센트로이드 데이터 처리"""
        # 대분류_센터로이드.csv 파일에서 센트로이드 데이터 활용
        try:
            self.centroids = self.data["centroid"].copy()
            print(f"📊 센트로이드 데이터 처리 완료: {len(self.centroids)}개")
        except Exception as e:
            print(f"⚠️ 기존 센트로이드 데이터 사용 실패: {str(e)}")
            
            # 추천결과_행단위.csv 파일에서 대분류와 권역별로 데이터 분류 및 센트로이드 계산
            try:
                recommend_data = self.data["recommend_result"]
                
                # 데이터 분류
                grouped_data = categorize_by_usage_type_and_region(recommend_data)
                
                # 센트로이드 계산
                available_cols = [col for col in self.norm_cols if col in recommend_data.columns]
                self.centroids = calculate_centroids(grouped_data, available_cols, method="median")
                
                print(f"📊 센트로이드 계산 완료: {len(self.centroids)}개")
            except Exception as e:
                print(f"⚠️ 센트로이드 계산 실패: {str(e)}")
                
                # 빈 센트로이드 생성
                self.centroids = pd.DataFrame(columns=["usage_type", "region"] + self.norm_cols)
    
    def recommend_by_query(self, 
                          query: str, 
                          algorithm: RecommendationAlgorithm = RecommendationAlgorithm.COSINE_SIMILARITY,
                          top_k: int = 10,
                          region: Optional[str] = None) -> RecommendationResponse:
        """주소 기반 추천"""
        if not query:
            return {"query": query, "timestamp": datetime.now(), "algorithm": algorithm, "count": 0, "items": []}
        
        # 주소 검색
        gas_df = self.data["gas_station"]
        filtered_df = gas_df[gas_df["주소"].astype(str).str.contains(query, na=False)]
        
        # 검색 결과가 없으면 행정구역으로 검색
        if filtered_df.empty:
            filtered_df = gas_df[gas_df["행정구역"].astype(str).str.contains(query, na=False)]
        
        # 여전히 결과가 없으면 빈 결과 반환
        if filtered_df.empty:
            return {"query": query, "timestamp": datetime.now(), "algorithm": algorithm, "count": 0, "items": []}
        
        # 권역 필터링
        if region:
            normalized_region = normalize_region(region)
            filtered_df = filtered_df[filtered_df["권역"] == normalized_region]
        
        # 센트로이드와 비교하여 추천
        if algorithm == RecommendationAlgorithm.POPULARITY:
            recommendations = self.recommend_by_popularity(filtered_df, top_k)
        elif algorithm == RecommendationAlgorithm.COLLABORATIVE:
            recommendations = self.recommend_by_collaborative_filtering(filtered_df, top_k)
        elif algorithm == RecommendationAlgorithm.PEARSON_CORRELATION:
            recommendations = self.recommend_by_pearson_correlation(filtered_df, top_k)
        elif algorithm == RecommendationAlgorithm.EUCLIDEAN_DISTANCE:
            recommendations = self.recommend_by_euclidean_distance(filtered_df, top_k)
        elif algorithm == RecommendationAlgorithm.AHP_TOPSIS:
            recommendations = self.recommend_by_ahp_topsis(filtered_df, top_k, region)
        else:  # 기본값: 코사인 유사도
            recommendations = self.recommend_by_cosine_similarity(filtered_df, top_k)
        
        # 결과 형식화
        return {
            "query": query,
            "timestamp": datetime.now(),
            "algorithm": algorithm,
            "count": len(recommendations),
            "items": recommendations
        }
    
    def recommend_by_popularity(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """인기도 기반 추천"""
        try:
            # 대분류 빈도수 계산 (추천결과_행단위.csv 파일 활용)
            recommend_df = self.data["recommend_result"]
            usage_type_counts = recommend_df["대분류"].value_counts().reset_index()
            usage_type_counts.columns = ["usage_type", "count"]
            
            # 상위 top_k개 선택
            top_usage_types = usage_type_counts.head(top_k)
            
            # 결과 형식화
            recommendations = []
            
            for i, (_, row) in enumerate(top_usage_types.iterrows()):
                usage_type = row["usage_type"]
                count = row["count"]
                
                # 첫 번째 주소 사용
                if len(df) > 0:
                    address_row = df.iloc[0]
                    address = address_row.get("주소", "")
                    admin_region = address_row.get("행정구역", "")
                    population = float(address_row.get("인구[명]", 0))
                    business_density = float(address_row.get("인구천명당사업체수", 0))
                    
                    recommendations.append({
                        "address": address,
                        "admin_region": admin_region,
                        "usage_type": usage_type,
                        "score": float(count / usage_type_counts["count"].max()),
                        "rank": i + 1,
                        "population": population,
                        "business_density": business_density,
                        "population_norm": float(address_row.get("인구[명]_norm", 0)),
                        "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
                        "station_name": address_row.get("상호명", ""),
                        "station_status": address_row.get("상태", ""),
                        "note": address_row.get("비고", "")
                    })
            
            return recommendations
        
        except Exception as e:
            print(f"⚠️ 인기도 기반 추천 실패: {str(e)}")
            return []
    
    def recommend_by_cosine_similarity(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """코사인 유사도 기반 추천"""
        try:
            if len(df) == 0 or len(self.centroids) == 0:
                return []
            
            # 첫 번째 주소의 벡터 추출
            address_row = df.iloc[0]
            
            # 필요한 벡터 확인
            available_norm_cols = [col for col in self.norm_cols if col in self.centroids.columns]
            
            if not available_norm_cols:
                return []
            
            # 주소 벡터 생성
            address_vector = np.array([address_row.get(col, 0) for col in available_norm_cols]).reshape(1, -1)
            
            # 센트로이드 벡터 생성
            centroids_vectors = self.centroids[available_norm_cols].values
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(address_vector, centroids_vectors)[0]
            
            # 유사도와 센트로이드 정보 결합
            similarity_df = self.centroids.copy()
            similarity_df["similarity"] = similarities
            
            # 유사도 기준 내림차순 정렬
            similarity_df = similarity_df.sort_values("similarity", ascending=False)
            
            # 상위 top_k개 선택
            top_centroids = similarity_df.head(top_k)
            
            # 결과 형식화
            recommendations = []
            
            for i, (_, centroid) in enumerate(top_centroids.iterrows()):
                recommendations.append({
                    "address": address_row.get("주소", ""),
                    "admin_region": address_row.get("행정구역", ""),
                    "usage_type": centroid.get("usage_type", ""),
                    "score": float(centroid.get("similarity", 0)),
                    "rank": i + 1,
                    "population": float(address_row.get("인구[명]", 0)),
                    "business_density": float(address_row.get("인구천명당사업체수", 0)),
                    "population_norm": float(address_row.get("인구[명]_norm", 0)),
                    "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
                    "traffic_norm": float(centroid.get("교통량_norm", 0)),
                    "tourism_norm": float(centroid.get("숙박업소(관광지수)_norm", 0)),
                    "land_price_norm": float(centroid.get("공시지가(토지단가)_norm", 0)),
                    "similarity": float(centroid.get("similarity", 0)),
                    "station_name": address_row.get("상호명", ""),
                    "station_status": address_row.get("상태", ""),
                    "note": address_row.get("비고", "")
                })
            
            return recommendations
        
        except Exception as e:
            print(f"⚠️ 코사인 유사도 기반 추천 실패: {str(e)}")
            return []
    
    def recommend_by_euclidean_distance(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """유클리드 거리 기반 추천"""
        try:
            if len(df) == 0 or len(self.centroids) == 0:
                return []
            
            # 첫 번째 주소의 벡터 추출
            address_row = df.iloc[0]
            
            # 필요한 벡터 확인
            available_norm_cols = [col for col in self.norm_cols if col in self.centroids.columns]
            
            if not available_norm_cols:
                return []
            
            # 주소 벡터 생성
            address_vector = np.array([address_row.get(col, 0) for col in available_norm_cols])
            
            # 센트로이드와 거리 계산
            distances = []
            
            for _, centroid in self.centroids.iterrows():
                # 센트로이드 벡터 생성
                centroid_vector = np.array([centroid.get(col, 0) for col in available_norm_cols])
                
                # 유클리드 거리 계산
                try:
                    distance = euclidean(address_vector, centroid_vector)
                except:
                    distance = float('inf')
                
                distances.append({
                    "usage_type": centroid.get("usage_type", ""),
                    "region": centroid.get("region", ""),
                    "distance": distance,
                    **{col: centroid.get(col, 0) for col in available_norm_cols}
                })
            
            # 거리 기준 오름차순 정렬
            sorted_distances = sorted(distances, key=lambda x: x["distance"])
            
            # 상위 top_k개 선택
            top_centroids = sorted_distances[:top_k]
            
            # 결과 형식화
            recommendations = []
            
            for i, centroid in enumerate(top_centroids):
                # 거리 점수 변환 (0~1, 가까울수록 1)
                max_distance = sorted_distances[-1]["distance"] if len(sorted_distances) > 0 else 1
                score = 1 - (centroid["distance"] / max_distance if max_distance > 0 else 0)
                
                recommendations.append({
                    "address": address_row.get("주소", ""),
                    "admin_region": address_row.get("행정구역", ""),
                    "usage_type": centroid.get("usage_type", ""),
                    "score": float(score),
                    "rank": i + 1,
                    "population": float(address_row.get("인구[명]", 0)),
                    "business_density": float(address_row.get("인구천명당사업체수", 0)),
                    "population_norm": float(address_row.get("인구[명]_norm", 0)),
                    "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
                    "traffic_norm": float(centroid.get("교통량_norm", 0)),
                    "tourism_norm": float(centroid.get("숙박업소(관광지수)_norm", 0)),
                    "land_price_norm": float(centroid.get("공시지가(토지단가)_norm", 0)),
                    "distance": float(centroid.get("distance", 0)),
                    "station_name": address_row.get("상호명", ""),
                    "station_status": address_row.get("상태", ""),
                    "note": address_row.get("비고", "")
                })
            
            return recommendations
        
        except Exception as e:
            print(f"⚠️ 유클리드 거리 기반 추천 실패: {str(e)}")
            return []
    
    def recommend_by_pearson_correlation(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """피어슨 상관계수 기반 추천"""
        try:
            if len(df) == 0 or len(self.centroids) == 0:
                return []
            
            # 첫 번째 주소의 벡터 추출
            address_row = df.iloc[0]
            
            # 필요한 벡터 확인
            available_norm_cols = [col for col in self.norm_cols if col in self.centroids.columns]
            
            if not available_norm_cols:
                return []
            
            # 주소 벡터 생성
            address_vector = np.array([address_row.get(col, 0) for col in available_norm_cols])
            
            # 센트로이드와 상관계수 계산
            correlations = []
            
            for _, centroid in self.centroids.iterrows():
                # 센트로이드 벡터 생성
                centroid_vector = np.array([centroid.get(col, 0) for col in available_norm_cols])
                
                # 피어슨 상관계수 계산
                try:
                    correlation, _ = pearsonr(address_vector, centroid_vector)
                    if math.isnan(correlation):
                        correlation = 0
                except:
                    correlation = 0
                
                correlations.append({
                    "usage_type": centroid.get("usage_type", ""),
                    "region": centroid.get("region", ""),
                    "correlation": correlation,
                    **{col: centroid.get(col, 0) for col in available_norm_cols}
                })
            
            # 상관계수 기준 내림차순 정렬
            sorted_correlations = sorted(correlations, key=lambda x: x["correlation"], reverse=True)
            
            # 상위 top_k개 선택
            top_centroids = sorted_correlations[:top_k]
            
            # 결과 형식화
            recommendations = []
            
            for i, centroid in enumerate(top_centroids):
                # 상관계수 점수 변환 (-1~1 -> 0~1)
                score = (centroid["correlation"] + 1) / 2
                
                recommendations.append({
                    "address": address_row.get("주소", ""),
                    "admin_region": address_row.get("행정구역", ""),
                    "usage_type": centroid.get("usage_type", ""),
                    "score": float(score),
                    "rank": i + 1,
                    "population": float(address_row.get("인구[명]", 0)),
                    "business_density": float(address_row.get("인구천명당사업체수", 0)),
                    "population_norm": float(address_row.get("인구[명]_norm", 0)),
                    "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
                    "traffic_norm": float(centroid.get("교통량_norm", 0)),
                    "tourism_norm": float(centroid.get("숙박업소(관광지수)_norm", 0)),
                    "land_price_norm": float(centroid.get("공시지가(토지단가)_norm", 0)),
                    "similarity": float(centroid.get("correlation", 0)),
                    "station_name": address_row.get("상호명", ""),
                    "station_status": address_row.get("상태", ""),
                    "note": address_row.get("비고", "")
                })
            
            return recommendations
        
        except Exception as e:
            print(f"⚠️ 피어슨 상관계수 기반 추천 실패: {str(e)}")
            return []
    
    def recommend_by_collaborative_filtering(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """협업 필터링 기반 추천"""
        try:
            # 추천결과_행단위.csv 파일 활용
            recommend_df = self.data["recommend_result"]
            
            if len(df) == 0 or len(recommend_df) == 0:
                return []
            
            # 첫 번째 주소 정보
            address_row = df.iloc[0]
            address_region = address_row.get("권역", "")
            
            # 주소의 권역과 일치하는 행 선택
            region_recommend = recommend_df[recommend_df["권역"] == address_region]
            
            if len(region_recommend) == 0:
                # 권역이 일치하지 않으면 전체 데이터 사용
                region_recommend = recommend_df
            
            # 대분류별 평균 추천 점수 계산
            usage_type_scores = region_recommend.groupby("대분류")["추천_대분류"].count().reset_index()
            usage_type_scores.columns = ["usage_type", "count"]
            
            # 점수 정규화
            total_count = usage_type_scores["count"].sum()
            usage_type_scores["score"] = usage_type_scores["count"] / total_count if total_count > 0 else 0
            
            # 내림차순 정렬
            usage_type_scores = usage_type_scores.sort_values("score", ascending=False)
            
            # 상위 top_k개 선택
            top_usage_types = usage_type_scores.head(top_k)
            
            # 결과 형식화
            recommendations = []
            
            for i, (_, row) in enumerate(top_usage_types.iterrows()):
                usage_type = row["usage_type"]
                score = row["score"]
                
                recommendations.append({
                    "address": address_row.get("주소", ""),
                    "admin_region": address_row.get("행정구역", ""),
                    "usage_type": usage_type,
                    "score": float(score),
                    "rank": i + 1,
                    "population": float(address_row.get("인구[명]", 0)),
                    "business_density": float(address_row.get("인구천명당사업체수", 0)),
                    "population_norm": float(address_row.get("인구[명]_norm", 0)),
                    "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
                    "station_name": address_row.get("상호명", ""),
                    "station_status": address_row.get("상태", ""),
                    "note": address_row.get("비고", "")
                })
            
            return recommendations
        
        except Exception as e:
            print(f"⚠️ 협업 필터링 기반 추천 실패: {str(e)}")
            return []
    
    def recommend_by_ahp_topsis(self, df: pd.DataFrame, top_k: int = 10, region: Optional[str] = None) -> List[Dict[str, Any]]:
        """권역 기반 AHP-TOPSIS 추천 알고리즘"""
        try:
            if len(df) == 0:
                return []
        
            # 첫 번째 주소의 정보 추출
            address_row = df.iloc[0]
            address_region = address_row.get("권역", "")
        
            # 권역이 없는 경우, extract_province 함수로 주소에서 추출 시도
            if not address_region:
                address = address_row.get("주소", "")
                address_region = extract_province(address)
        
            # 권역 정규화
            if region:
                # 사용자가 지정한 권역 사용
                address_region = normalize_region(region)
            elif address_region:
                # 추출된 권역 정규화
                address_region = normalize_region(address_region)
            else:
                # 권역을 찾을 수 없는 경우 기본값으로 "전라북도" 사용
                address_region = "전라북도"
        
            # 추천 결과 데이터 가져오기
            recommend_df = self.data.get("recommend_result", pd.DataFrame())
        
            # 권역별 데이터 필터링
            region_df = recommend_df[recommend_df["권역"] == address_region] if "권역" in recommend_df.columns else recommend_df
        
            if len(region_df) == 0:
                return []
        
            # 1. AHP 가중치 정의 (4개 지표)
            # 여기서는 고정된 가중치를 사용하지만, 실제로는 쌍대비교행렬로부터 계산할 수 있습니다.
            weights = {
                "인구[명]_norm": 0.30,           # 인구밀도
                "교통량_norm": 0.25,             # 교통량
                "숙박업소(관광지수)_norm": 0.20,  # 관광지수
                "상권밀집도(비율)_norm": 0.25     # 상권밀집도
            }
        
            # 사용 가능한 특징 컬럼 확인
            available_cols = [col for col in weights.keys() if col in address_row.index]
        
            if not available_cols:
                return []
        
            # 2. 각 대안(용도 유형)별 지표 중위값 계산
            usage_types = region_df["대분류"].unique() if "대분류" in region_df.columns else []

            if len(usage_types) == 0:
                # 대안이 없는 경우, 기본 대안 사용
                usage_types = [
                    "근린생활시설", "공동주택", "자동차관련시설", 
                    "판매시설", "업무시설", "숙박시설",
                    "공장", "가설건축", "기타"
                ]
        
            # 3. 결정 행렬 생성
            decision_matrix = {}
        
            for usage_type in usage_types:
                # 해당 용도 유형과 권역의 데이터 필터링
                type_df = region_df[region_df["대분류"] == usage_type] if "대분류" in region_df.columns else pd.DataFrame()
            
                if len(type_df) > 0:
                    # 중위값 계산
                    medians = {}
                    for col in available_cols:
                        if col in type_df.columns:
                            medians[col] = type_df[col].median()
                        else:
                            medians[col] = 0.5  # 기본값

                    decision_matrix[usage_type] = medians
                else:
                    # 데이터가 없는 경우 기본값 사용
                    decision_matrix[usage_type] = {col: 0.5 for col in available_cols}
        
            # 4. 대상 주유소의 지표 값 추출
            site_values = {}
            for col in available_cols:
                site_values[col] = float(address_row.get(col, 0))
        
            # 5. TOPSIS 알고리즘 적용
            # 5.1. 유사도 점수 행렬 생성 (대상 주유소와 각 용도 유형의 중위값 간 유사도)
            similarity_matrix = {}
        
            for usage_type, medians in decision_matrix.items():
                distances = {}

                for col in available_cols:
                    # 절대 거리 계산
                    distance = abs(site_values[col] - medians[col])
                    distances[col] = 1 - distance  # 유사도로 변환 (값이 클수록 유사)

                similarity_matrix[usage_type] = distances

            # 5.2. 정규화 및 가중치 적용
            weighted_matrix = {}
        
            for usage_type, similarities in similarity_matrix.items():
                weighted = {}

                for col in available_cols:
                    weighted[col] = weights.get(col, 0) * similarities[col]

                weighted_matrix[usage_type] = weighted

            # 5.3. 이상해 및 반대해 계산
            ideal_positive = {}
            ideal_negative = {}
        
            for col in available_cols:
                max_val = max(weighted_matrix[ut][col] for ut in weighted_matrix)
                min_val = min(weighted_matrix[ut][col] for ut in weighted_matrix)

                ideal_positive[col] = max_val
                ideal_negative[col] = min_val
        
            # 5.4. 거리 계산
            distances_positive = {}
            distances_negative = {}
        
            for usage_type, weighted in weighted_matrix.items():
                # 이상해와의 거리
                dist_pos = sum((weighted[col] - ideal_positive[col]) ** 2 for col in available_cols) ** 0.5
                # 반대해와의 거리
                dist_neg = sum((weighted[col] - ideal_negative[col]) ** 2 for col in available_cols) ** 0.5

                distances_positive[usage_type] = dist_pos
                distances_negative[usage_type] = dist_neg
        
            # 5.5. 상대 근접도 계산
            closeness = {}

            for usage_type in weighted_matrix:
                d_pos = distances_positive[usage_type]
                d_neg = distances_negative[usage_type]
            
                # 0으로 나누기 방지
                if d_pos + d_neg == 0:
                    closeness[usage_type] = 0
                else:
                    closeness[usage_type] = d_neg / (d_pos + d_neg)

            # 6. 결과 정렬 및 상위 추천 반환
            sorted_results = sorted(
                [(usage_type, score) for usage_type, score in closeness.items()],
                key=lambda x: x[1],
                reverse=True
            )
        
            # 상위 top_k개 선택
            top_results = sorted_results[:top_k]
        
            # 7. 결과 형식화
            recommendations = []

            for i, (usage_type, score) in enumerate(top_results):
                recommendations.append({
                    "address": address_row.get("주소", ""),
                    "admin_region": address_row.get("행정구역", ""),
                    "usage_type": usage_type,
                    "score": float(score),
                    "rank": i + 1,
                    "population": float(address_row.get("인구[명]", 0)),
                    "business_density": float(address_row.get("인구천명당사업체수", 0)),
                    "population_norm": float(address_row.get("인구[명]_norm", 0)),
                    "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
                    "traffic_norm": float(address_row.get("교통량_norm", 0) if "교통량_norm" in address_row else 0),
                    "tourism_norm": float(address_row.get("숙박업소(관광지수)_norm", 0) if "숙박업소(관광지수)_norm" in address_row else 0),
                    "land_price_norm": float(address_row.get("공시지가(토지단가)_norm", 0) if "공시지가(토지단가)_norm" in address_row else 0),
                    "ahp_weights": weights,  # AHP 가중치 정보 추가
                    "region": address_region,  # 권역 정보 추가
                    "station_name": address_row.get("상호명", ""),
                    "station_status": address_row.get("상태", ""),
                    "note": address_row.get("비고", "")
                })

            return recommendations

        except Exception as e:
            print(f"⚠️ 권역 기반 AHP-TOPSIS 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []