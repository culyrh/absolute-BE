"""
주유소 정보 관련 API 엔드포인트
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Path
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse

from app.api.dependencies import get_geo_service
from app.services.geo_service import GeoService
from app.schemas.gas_station import GasStationList, GasStationResponse


router = APIRouter(
    prefix="/api/stations",
    tags=["gas_stations"],
    responses={404: {"description": "Not found"}},
)


@router.get("/region/{code}", response_model=GasStationList)
async def get_stations_by_region(
    code: str = Path(..., description="지역 코드"),
    limit: int = Query(100, ge=1, le=1000, description="반환할 결과 수"),
    service: GeoService = Depends(get_geo_service),
):
    """
    지역별 주유소 목록 API
    
    - **code**: 지역 코드 (필수)
    - **limit**: 반환할 결과 수 (기본값: 100, 최대: 1000)
    """
    try:
        # 행정구역으로 검색
        result = service.search_by_region(code, limit)
        
        # 캐싱 헤더 설정 (1시간)
        headers = {"Cache-Control": "public, max-age=3600"}
        
        return JSONResponse(
            content={"count": len(result), "items": result},
            headers=headers
        )
    except Exception as e:
        print(f"지역별 주유소 목록 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"지역별 주유소 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/map", response_model=GasStationList)
async def get_stations_in_map(
    lat1: float = Query(..., description="위도 최소값"),
    lng1: float = Query(..., description="경도 최소값"),
    lat2: float = Query(..., description="위도 최대값"),
    lng2: float = Query(..., description="경도 최대값"),
    limit: int = Query(100, ge=1, le=1000, description="반환할 결과 수"),
    service: GeoService = Depends(get_geo_service),
):
    """
    지도 범위 내 주유소 API
    
    - **lat1**: 위도 최소값 (필수)
    - **lng1**: 경도 최소값 (필수)
    - **lat2**: 위도 최대값 (필수)
    - **lng2**: 경도 최대값 (필수)
    - **limit**: 반환할 결과 수 (기본값: 100, 최대: 1000)
    """
    try:
        # 폐휴업 주유소 데이터에서 좌표로 검색
        gas_df = service.data.get("closed_gas_station", None)
        
        # 좌표 데이터가 없는 경우 빈 결과 반환
        if gas_df is None or "위도" not in gas_df.columns or "경도" not in gas_df.columns:
            return JSONResponse(content={"count": 0, "items": []})
        
        # 좌표 범위 내 데이터 필터링
        filtered_df = gas_df[
            (gas_df["위도"] >= lat1) & 
            (gas_df["위도"] <= lat2) & 
            (gas_df["경도"] >= lng1) & 
            (gas_df["경도"] <= lng2)
        ]
        
        # 결과 형식화
        result = filtered_df.head(limit).to_dict("records")
        
        # 캐싱 헤더 설정 (5분)
        headers = {"Cache-Control": "public, max-age=300"}
        
        return JSONResponse(
            content={"count": len(result), "items": result},
            headers=headers
        )
    except Exception as e:
        print(f"지도 범위 내 주유소 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"지도 범위 내 주유소 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/search", response_model=GasStationList)
async def search_stations(
    query: str = Query(..., description="주소 검색어"),
    limit: int = Query(100, ge=1, le=1000, description="반환할 결과 수"),
    service: GeoService = Depends(get_geo_service),
):
    """
    주소 기반 검색 API
    
    - **query**: 주소 검색어 (필수)
    - **limit**: 반환할 결과 수 (기본값: 100, 최대: 1000)
    """
    try:
        # 주소로 검색
        result = service.search_by_address(query, limit)
        return {"count": len(result), "items": result}
    except Exception as e:
        print(f"주소 기반 검색 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"주소 기반 검색 중 오류가 발생했습니다: {str(e)}")


@router.get("/{id}", response_model=GasStationResponse)
async def get_station_detail(
    id: int = Path(..., description="주유소 ID"),
    service: GeoService = Depends(get_geo_service),
):
    """
    개별 주유소 상세 정보 API
    
    - **id**: 주유소 ID (필수)
    """
    try:
        station = service.get_station_by_id(id)
        
        if not station:
            raise HTTPException(status_code=404, detail=f"ID가 {id}인 주유소를 찾을 수 없습니다.")
        
        # 캐싱 헤더 설정 (1일)
        headers = {"Cache-Control": "public, max-age=86400"}
        
        return JSONResponse(
            content=station,
            headers=headers
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"주유소 상세 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"주유소 상세 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/cases", response_model=Dict[str, Any])
async def get_station_cases():
    """
    활용 사례 카드 API
    
    폐주유소의 다양한 활용 사례 정보를 카드 형태로 제공합니다.
    """
    try:
        # 대분류 정보 활용한 활용 사례 카드
        cases = [
            {
                "id": 1,
                "title": "근린생활시설",
                "description": "일상생활에 필요한 서비스를 제공하는 시설로 활용",
                "image_url": "/assets/cases/convenience.jpg"
            },
            {
                "id": 2,
                "title": "공동주택",
                "description": "주거 공간으로 재활용하여 주택 공급에 기여",
                "image_url": "/assets/cases/housing.jpg"
            },
            {
                "id": 3,
                "title": "자동차관련시설",
                "description": "전기차 충전소나 정비소로 전환하여 활용",
                "image_url": "/assets/cases/automotive.jpg"
            },
            {
                "id": 4,
                "title": "판매시설",
                "description": "소매점이나 마켓으로 활용하여 지역 상권 활성화",
                "image_url": "/assets/cases/retail.jpg"
            },
            {
                "id": 5,
                "title": "업무시설",
                "description": "코워킹 스페이스나 사무실로 활용",
                "image_url": "/assets/cases/office.jpg"
            }
        ]
        
        # 캐싱 헤더 설정 (1일)
        headers = {"Cache-Control": "public, max-age=86400"}
        
        return JSONResponse(
            content={"count": len(cases), "items": cases},
            headers=headers
        )
    except Exception as e:
        print(f"활용 사례 카드 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"활용 사례 카드 조회 중 오류가 발생했습니다: {str(e)}")
