"""
폐주유소 활용 추천 API 서버
메인 애플리케이션 진입점
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import importlib.util
import sys
import os

# 실제 모듈 경로 확인
if os.path.exists("app/api/endpoints/recommend.py"):
    # 기존 프로젝트 구조 사용
    from app.api.endpoints import recommend, stations, usage_types, ml_recommend
    from app.core.config import settings
    
    # FastAPI 애플리케이션 생성
    app = FastAPI(
        title=settings.APP_NAME,
        description="폐주유소 활용 용도 추천 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 라우터 등록
    app.include_router(recommend.router)
    app.include_router(stations.router)
    app.include_router(usage_types.router)
    app.include_router(ml_recommend.router)    #  ML기반추천 추가했습니다
else:
    # 단순화된 구조 사용
    import recommend
    import stations
    import s3
    import ml_recommend # ml 추가했습니다

    # FastAPI 애플리케이션 생성
    app = FastAPI(
        title="폐주유소 활용 추천 API",
        description="폐주유소 활용 용도 추천 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 라우터 등록
    app.include_router(recommend.router)
    app.include_router(stations.router)
    app.include_router(s3.router)
    app.include_router(ml_recommend.router)    # ML기반추천 추가

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 오리진 허용 (실제 운영 환경에서는 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 경로 정의
@app.get("/")
def read_root():
    """API 서버 정보 반환"""
    return {
        "name": "폐주유소 활용 추천 API",
        "version": "1.0.0",
        "description": "폐주유소 활용 용도 추천 API",
        "endpoints": {
            "api/stations/region/{code}": "지역별 주유소 목록",
            "api/stations/map": "지도 범위 내 주유소",
            "api/stations/search": "주소 기반 검색",
            "api/stations/{id}": "개별 주유소 상세 정보",
            "api/stations/cases": "활용 사례 카드",
            "api/recommend": "추천 시스템",
            "api/ml-recommend": "ML 기반 추천 시스템",  # 추가했습니다
            "api/s3/presigned": "S3 업로드 URL 발급 (예정)"
        }
    }

# 애플리케이션 실행 (개발용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)