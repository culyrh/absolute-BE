"""
추천 관련 API 엔드포인트
"""

from fastapi import APIRouter, Depends, Body, HTTPException
from typing import Dict, Any, Optional

from app.api.dependencies import get_recommendation_service
from app.services.recommend_service import RecommendationService


# 라우터 생성
router = APIRouter(
    prefix="/api/recommend",
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)


# 추천 시스템 API (POST)
@router.post("", response_model=Dict[str, Any])
async def recommend_usage(
    location: Dict[str, Any] = Body(..., description="위치 정보 (주소, 좌표 등)"),
    options: Optional[Dict[str, Any]] = Body(None, description="추가 옵션 (권역, 알고리즘 등)"),
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    추천 시스템 API (POST)
    
    위치 정보를 기반으로 적합한 활용 방안을 추천합니다.
    
    - **location**: 위치 정보 (필수)
      - address: 주소
      - coordinates: 좌표 (위도, 경도)
    - **options**: 추가 옵션 (선택)
      - region: 특정 권역 필터
      - algorithm: 추천 알고리즘 유형
      - top_k: 반환할 결과 수 (기본값: 10, 최대: 100)
    
    **Response**:
    - count: 추천 결과 수
    - items: 추천 결과 목록
      - type: 활용 유형 (대분류)
      - score: 추천 점수 (0~1)
      - description: 추천 설명
    """
    try:
        # 주소 추출
        address = location.get("address", "")
        
        # 좌표 추출
        coordinates = location.get("coordinates", {})
        lat = coordinates.get("lat", None)
        lng = coordinates.get("lng", None)
        
        # 옵션 추출
        region = options.get("region", None) if options else None
        algorithm_name = options.get("algorithm", "cosine_similarity") if options else "cosine_similarity"
        top_k = min(100, max(1, options.get("top_k", 10))) if options else 10
        
        # 알고리즘 유형 변환
        from app.schemas.recommendation import RecommendationAlgorithm
        algorithm = RecommendationAlgorithm.COSINE_SIMILARITY
        
        if algorithm_name == "popularity":
            algorithm = RecommendationAlgorithm.POPULARITY
        elif algorithm_name == "collaborative":
            algorithm = RecommendationAlgorithm.COLLABORATIVE
        elif algorithm_name == "pearson_correlation":
            algorithm = RecommendationAlgorithm.PEARSON_CORRELATION
        elif algorithm_name == "euclidean_distance":
            algorithm = RecommendationAlgorithm.EUCLIDEAN_DISTANCE
        elif algorithm_name == "ahp_topsis":
            algorithm = RecommendationAlgorithm.AHP_TOPSIS
        
        # 주소 기반 추천 (기본)
        if address:
            result = service.recommend_by_query(address, algorithm, top_k, region)
            return result
        
        # 좌표 기반 추천 (주소가 없고 좌표가 있는 경우)
        elif lat is not None and lng is not None:
            # 여기에 좌표 기반 추천 로직 구현 (현재는 더미 데이터)
            # 실제 구현에서는 좌표에 가장 가까운 주유소를 찾아서 해당 주소로 추천 필요
            dummy_result = {
                "query": f"{lat},{lng}",
                "timestamp": "2025-10-28T12:00:00Z",
                "algorithm": algorithm_name,
                "count": 5,
                "items": [
                    {
                        "type": "근린생활시설",
                        "score": 0.95,
                        "description": "주변 상권과 인구 밀도를 고려할 때 근린생활시설로 활용하는 것이 가장 적합합니다."
                    },
                    {
                        "type": "자동차관련시설",
                        "score": 0.85,
                        "description": "교통량이 많은 지역으로 자동차 관련 시설로 활용할 수 있습니다."
                    },
                    {
                        "type": "판매시설",
                        "score": 0.75,
                        "description": "판매시설로 활용하여 지역 상권 활성화에 기여할 수 있습니다."
                    },
                    {
                        "type": "업무시설",
                        "score": 0.65,
                        "description": "업무시설로 활용하여 지역 일자리 창출에 기여할 수 있습니다."
                    },
                    {
                        "type": "공동주택",
                        "score": 0.55,
                        "description": "주거 공간으로 활용하여 주택 공급에 기여할 수 있습니다."
                    }
                ]
            }
            return dummy_result
        
        # 주소와 좌표 모두 없는 경우
        else:
            raise HTTPException(status_code=400, detail="주소 또는 좌표 정보가 필요합니다.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️ 추천 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"추천 처리 중 오류가 발생했습니다: {str(e)}")