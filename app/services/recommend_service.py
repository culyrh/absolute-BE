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
