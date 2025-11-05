from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query

from app.services.ml_location_recommender import MLLocationRecommender

router = APIRouter(
    prefix="/api/ml-recommend",
    tags=["ml_recommendations"],
)

recommender = MLLocationRecommender()
try:
    recommender.train()
except Exception as e:
    print(f"MLLocationRecommender 초기 학습 실패: {e}")


@router.get("", response_model=Dict[str, Any])
async def ml_recommend(
    keyword: str = Query(
        ...,
        description="주유소 상호명 또는 주소에 포함된 검색어 (예: '상무제일주유소')",
    ),
    top_n: int = Query(3, ge=1, le=10),
) -> Dict[str, Any]:
    try:
        result = recommender.recommend_for_station(keyword, top_n=top_n)
        return result
    except Exception as e:
        print(f"ml_recommend API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"ML 추천 중 오류 발생: {e}")
