"""
폐주유소 활용 추천 API 서버
메인 애플리케이션 진입점
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import stations, usage_types, ml_recommend
from app.core.config import get_settings

from dotenv import load_dotenv
load_dotenv()

settings = get_settings()

# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.APP_NAME,
    description="폐주유소 활용 용도 추천 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 라우터 등록
app.include_router(stations.router)
app.include_router(usage_types.router)
app.include_router(ml_recommend.router)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 정보
@app.get("/")
def read_root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "description": "폐주유소 활용 용도 추천 API",
        "endpoints": {
            "api/stations/region/{code}": "지역별 주유소 목록",
            "api/stations/map": "지도 범위 내 주유소",
            "api/stations/search": "주소 기반 검색",
            "api/stations/{id}": "개별 주유소 상세 정보",
            "api/stations/cases": "활용 사례 카드",
            "api/ml-recommend": "ML 기반 추천 시스템",
        },
    }

# 개발 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
