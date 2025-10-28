"""
애플리케이션 설정 및 환경 변수 관리
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # 기본 정보
    APP_NAME: str = "폐주유소 활용 추천 API"
    API_PREFIX: str = "/api"
    
    # 데이터 경로
    DATA_DIR: str = "data"
    GAS_STATION_FILE: str = "jeonju_gas_station.csv"
    POPULATION_FILE: str = "전국인구수.xlsx"
    BUSINESS_FILE: str = "전국1000명당사업체수.xlsx"
    CENTER_FILE: str = "대분류_센터로이드.csv"
    RECOMMEND_RESULT_FILE: str = "추천결과_행단위.csv"
    CLOSED_GAS_STATION_FILE: str = "산업통상자원부_폐휴업주유소_좌표추가_kakao.csv"
    
    # CORS 설정
    CORS_ORIGINS: List[str] = ["*"]
    
    # 데이터베이스 설정 (향후 구현)
    DATABASE_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """캐시된 설정 인스턴스 반환"""
    return Settings()


# 설정 인스턴스 생성
settings = get_settings()

# 프로젝트 기본 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / settings.DATA_DIR
