"""
추천 관련 API 스키마
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class RecommendationAlgorithm(str, Enum):
    """추천 알고리즘 유형"""
    POPULARITY = "popularity"  # 인기도 기반
    COLLABORATIVE = "collaborative"  # 협업 필터링
    COSINE_SIMILARITY = "cosine_similarity"  # 코사인 유사도
    PEARSON_CORRELATION = "pearson_correlation"  # 피어슨 상관계수
    EUCLIDEAN_DISTANCE = "euclidean_distance"  # 유클리드 거리


class RecommendationRequest(BaseModel):
    """추천 요청 스키마"""
    query: str = Field(..., description="검색어 (주소 또는 관할주소)")
    algorithm: RecommendationAlgorithm = Field(
        default=RecommendationAlgorithm.COSINE_SIMILARITY, 
        description="추천 알고리즘 유형"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="반환할 결과 수")
    region: Optional[str] = Field(None, description="특정 권역 필터")


class RecommendationItemBase(BaseModel):
    """추천 항목 기본 스키마"""
    address: str = Field(..., description="주소")
    admin_region: Optional[str] = Field(None, description="행정구역")
    usage_type: str = Field(..., description="추천 용도 유형 (대분류)")
    score: float = Field(..., description="추천 점수")
    rank: int = Field(..., description="순위")
    

class RecommendationItemDetail(RecommendationItemBase):
    """추천 항목 상세 스키마"""
    population: Optional[float] = Field(None, description="인구수")
    business_density: Optional[float] = Field(None, description="사업체 밀집도")
    population_norm: Optional[float] = Field(None, description="인구수 정규화 값")
    business_density_norm: Optional[float] = Field(None, description="사업체 밀집도 정규화 값")
    traffic_norm: Optional[float] = Field(None, description="교통량 정규화 값")
    tourism_norm: Optional[float] = Field(None, description="숙박업소(관광지수) 정규화 값")
    land_price_norm: Optional[float] = Field(None, description="공시지가 정규화 값")
    distance: Optional[float] = Field(None, description="유클리드 거리")
    similarity: Optional[float] = Field(None, description="유사도")
    station_name: Optional[str] = Field(None, description="주유소명")
    station_status: Optional[str] = Field(None, description="주유소 상태")
    note: Optional[str] = Field(None, description="비고")


class RecommendationResponse(BaseModel):
    """추천 응답 스키마"""
    query: str = Field(..., description="검색어")
    timestamp: datetime = Field(..., description="추천 시간")
    algorithm: RecommendationAlgorithm = Field(..., description="사용된 알고리즘")
    count: int = Field(..., description="결과 수")
    items: List[RecommendationItemDetail] = Field(..., description="추천 항목 목록")


class RecommendationStats(BaseModel):
    """추천 통계 스키마"""
    total_recommendations: int = Field(..., description="전체 추천 수")
    top_queries: List[Dict[str, Any]] = Field(..., description="인기 검색어")
    top_regions: List[Dict[str, Any]] = Field(..., description="인기 지역")
    top_usage_types: List[Dict[str, Any]] = Field(..., description="인기 용도 유형")
    algorithm_usage: Dict[str, int] = Field(..., description="알고리즘 사용 통계")
