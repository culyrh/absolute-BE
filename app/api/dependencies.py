"""
API 공통 의존성
"""

from fastapi import Depends
from app.services.recommend_service import RecommendationService
from app.services.geo_service import GeoService


def get_recommendation_service() -> RecommendationService:
    """추천 서비스 의존성"""
    return RecommendationService()


def get_geo_service() -> GeoService:
    """지리 정보 서비스 의존성"""
    return GeoService()
