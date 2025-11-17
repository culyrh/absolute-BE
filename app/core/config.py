"""
애플리케이션 설정 및 환경 변수 관리
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    """애플리케이션 설정 클래스"""
    
    # 기본 정보
    APP_NAME: str = "폐주유소 활용 추천 API"
    API_PREFIX: str = "/api"
    
    # 데이터 경로
    GAS_STATION_FILE: str = str(DATA_DIR / "station.csv")
    POPULATION_FILE: str = str(DATA_DIR / "전국인구수_행정동별.csv")
    BUSINESS_FILE: str = str(DATA_DIR / "전국1000명당사업체수_행정동별.csv")
    CENTER_FILE: str = str(DATA_DIR / "대분류_센터로이드.csv")
    RECOMMEND_RESULT_FILE: str = str(DATA_DIR / "추천결과_행단위.csv")
    INTEGRATED_DATA_FILE: str = str(DATA_DIR / "train.csv")
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    # DB (사용 안 하면 None)
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_db: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
