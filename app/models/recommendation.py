"""
추천 관련 데이터 모델
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum


class RecommendationAlgorithm(str, Enum):
    """추천 알고리즘 유형"""
    POPULARITY = "popularity"  # 인기도 기반
    COLLABORATIVE = "collaborative"  # 협업 필터링
    COSINE_SIMILARITY = "cosine_similarity"  # 코사인 유사도
    PEARSON_CORRELATION = "pearson_correlation"  # 피어슨 상관계수
    EUCLIDEAN_DISTANCE = "euclidean_distance"  # 유클리드 거리


class RecommendationBase(BaseModel):
    """추천 기본 모델"""
    address: str
    admin_region: Optional[str] = None
    population: Optional[float] = None
    traffic: Optional[float] = None
    tourism: Optional[float] = None
    business_density: Optional[float] = None
    land_price: Optional[float] = None
    population_norm: Optional[float] = None
    traffic_norm: Optional[float] = None
    tourism_norm: Optional[float] = None
    business_density_norm: Optional[float] = None
    land_price_norm: Optional[float] = None


class RecommendationItem(RecommendationBase):
    """추천 항목 모델"""
    usage_type: str
    score: float
    rank: int
    distance: Optional[float] = None
    similarity: Optional[float] = None
    algorithm: RecommendationAlgorithm


class Recommendation(BaseModel):
    """추천 결과 모델"""
    query: str
    timestamp: str
    algorithm: RecommendationAlgorithm
    items: List[RecommendationItem]


class RecommendationRequest(BaseModel):
    """추천 요청 모델"""
    query: str
    algorithm: Optional[RecommendationAlgorithm] = RecommendationAlgorithm.COSINE_SIMILARITY
    top_k: Optional[int] = 10
    region: Optional[str] = None


class RecommendationVector(BaseModel):
    """추천 벡터 모델"""
    usage_type: str
    region: str
    population: Optional[float] = None
    traffic: Optional[float] = None
    tourism: Optional[float] = None
    business_density: Optional[float] = None
    land_price: Optional[float] = None
    population_norm: Optional[float] = None
    traffic_norm: Optional[float] = None
    tourism_norm: Optional[float] = None
    business_density_norm: Optional[float] = None
    land_price_norm: Optional[float] = None
