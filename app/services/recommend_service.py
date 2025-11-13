"""
추천 시스템 서비스
알고리즘 로직을 app/comparison/algorithms/에 분리
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.utils.data_loader import load_all_data
from app.utils.preprocessing import (
    preprocess_gas_station_data, merge_with_stats, normalize_features,
    normalize_region
)
from app.schemas.recommendation import RecommendationAlgorithm, RecommendationResponse

# 알고리즘 클래스 임포트
from app.comparison.algorithms.cosine_similarity import CosineSimilarityAlgorithm
from app.comparison.algorithms.euclidean_distance import EuclideanDistanceAlgorithm
from app.comparison.algorithms.pearson_correlation import PearsonCorrelationAlgorithm
from app.comparison.algorithms.popularity import PopularityAlgorithm
from app.comparison.algorithms.collaborative import CollaborativeAlgorithm
from app.comparison.algorithms.ahp_topsis import AHPTopsisAlgorithm


class RecommendationService:
    """추천 시스템 서비스 - 알고리즘 객체 관리 및 호출만 담당"""
    
    def __init__(self):
        self.data = None
        self.centroids = None
        self.feature_cols = ["인구[명]", "교통량", "숙박업소(관광지수)", "상권밀집도(비율)", "공시지가(토지단가)"]
        self.norm_cols = [f"{col}_norm" for col in self.feature_cols]
        self.algorithms = {}  # 알고리즘 객체 캐싱
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
        
        # 알고리즘 객체 초기화
        self._initialize_algorithms()
        
        print("✅ 추천 서비스 초기화 완료")
    
    def process_centroids(self):
        """센트로이드 데이터 처리"""
        try:
            self.centroids = self.data["centroid"].copy()
        
            # 대분류 컬럼을 usage_type으로 변환
            if "대분류" in self.centroids.columns:
                self.centroids = self.centroids.rename(columns={"대분류": "usage_type"})
                print("✅ 센트로이드 컬럼명 변환: 대분류 → usage_type")
        
            print(f"📊 센트로이드 데이터 처리 완료: {len(self.centroids)}개")
            if "usage_type" in self.centroids.columns:
                print(f"📊 용도 유형: {self.centroids['usage_type'].unique().tolist()}")
            
        except Exception as e:
            print(f"⚠️ 센트로이드 데이터 로드 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            # 빈 센트로이드 생성
            self.centroids = pd.DataFrame(columns=["usage_type", "region"] + self.norm_cols)
    
    
    def _initialize_algorithms(self):
        """모든 알고리즘 객체 초기화"""
        try:
            train_data = self.data.get("recommend_result", pd.DataFrame())
            
            self.algorithms = {
                RecommendationAlgorithm.COSINE_SIMILARITY: CosineSimilarityAlgorithm(
                    self.centroids, self.norm_cols
                ),
                RecommendationAlgorithm.EUCLIDEAN_DISTANCE: EuclideanDistanceAlgorithm(
                    self.centroids, self.norm_cols
                ),
                RecommendationAlgorithm.PEARSON_CORRELATION: PearsonCorrelationAlgorithm(
                    self.centroids, self.norm_cols
                ),
                RecommendationAlgorithm.POPULARITY: PopularityAlgorithm(
                    self.centroids, self.norm_cols, train_data
                ),
                RecommendationAlgorithm.COLLABORATIVE: CollaborativeAlgorithm(
                    self.centroids, self.norm_cols, train_data
                ),
                RecommendationAlgorithm.AHP_TOPSIS: AHPTopsisAlgorithm(
                    self.centroids, self.norm_cols, train_data
                ),
            }
            print(f"✅ {len(self.algorithms)}개 알고리즘 초기화 완료")
        except Exception as e:
            print(f"⚠️ 알고리즘 초기화 실패: {str(e)}")
            self.algorithms = {}
    
    def recommend_by_query(self, 
                          query: str, 
                          algorithm: RecommendationAlgorithm = RecommendationAlgorithm.COSINE_SIMILARITY,
                          top_k: int = 10,
                          region: Optional[str] = None) -> RecommendationResponse:
        """
        주소 기반 추천
        
        Args:
            query: 검색 쿼리 (주소)
            algorithm: 사용할 알고리즘
            top_k: 반환할 결과 수
            region: 권역 필터 (선택)
            
        Returns:
            추천 결과
        """
        if not query:
            return {
                "query": query, 
                "timestamp": datetime.now(), 
                "algorithm": algorithm, 
                "count": 0, 
                "items": []
            }
        
        # 주소 검색
        gas_df = self.data["gas_station"]
        filtered_df = gas_df[gas_df["주소"].astype(str).str.contains(query, na=False)]
        
        # 검색 결과가 없으면 행정구역으로 검색
        if filtered_df.empty:
            filtered_df = gas_df[gas_df["행정구역"].astype(str).str.contains(query, na=False)]
        
        # 여전히 결과가 없으면 빈 결과 반환
        if filtered_df.empty:
            return {
                "query": query, 
                "timestamp": datetime.now(), 
                "algorithm": algorithm, 
                "count": 0, 
                "items": []
            }
        
        # 권역 필터링
        if region:
            normalized_region = normalize_region(region)
            filtered_df = filtered_df[filtered_df["권역"] == normalized_region]
        
        # 알고리즘 선택 및 실행
        algorithm_obj = self.algorithms.get(algorithm)
        
        if algorithm_obj is None:
            # 기본 알고리즘(코사인 유사도) 사용
            algorithm = RecommendationAlgorithm.COSINE_SIMILARITY
            algorithm_obj = self.algorithms.get(algorithm)
        
        # 추천 실행
        try:
            recommendations = algorithm_obj.recommend(filtered_df, top_k=top_k)
        except Exception as e:
            print(f"⚠️ 추천 실행 중 오류: {str(e)}")
            recommendations = []
        
        # 결과 형식화
        return {
            "query": query,
            "timestamp": datetime.now(),
            "algorithm": algorithm,
            "count": len(recommendations),
            "items": recommendations
        }
    
    def get_available_algorithms(self) -> List[str]:
        """사용 가능한 알고리즘 목록 반환"""
        return [algo.value for algo in self.algorithms.keys()]
    
    def get_algorithm_info(self, algorithm: RecommendationAlgorithm) -> Dict[str, str]:
        """특정 알고리즘의 정보 반환"""
        algorithm_obj = self.algorithms.get(algorithm)
        
        if algorithm_obj is None:
            return {
                "name": "Unknown",
                "description": "알고리즘을 찾을 수 없습니다."
            }
        
        return {
            "name": algorithm_obj.name,
            "description": algorithm_obj.description
        }


def get_recommendation_service() -> RecommendationService:
    # 매번 새로 만드는 게 부담되면, 싱글톤처럼 캐싱해도 됨
    if not hasattr(get_recommendation_service, "_instance"):
        get_recommendation_service._instance = RecommendationService()
    return get_recommendation_service._instance
