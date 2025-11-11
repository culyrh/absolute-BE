"""
주유소 정보 관련 API 엔드포인트
"""

import folium
from fastapi import APIRouter, Depends, Query, HTTPException, Path
from typing import Optional, List, Dict, Any
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_geo_service
from app.services.geo_service import GeoService
from app.schemas.gas_station import GasStationList, GasStationResponse
from app.services.parcel_service import get_parcel_service
from app.services.recommend_service import RecommendationService, get_recommendation_service


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


@router.get("/{id}/report", response_class=HTMLResponse)
async def generate_station_report(
    id: int = Path(..., description="주유소 ID"),
    service: GeoService = Depends(get_geo_service),
    recommend_service: RecommendationService = Depends(get_recommendation_service)
):
    """
    주유소 입지 분석 보고서 (지적도 포함)
    
    Returns:
        HTML 보고서
    """
    try:
        # 1. 주유소 정보
        station = service.get_station_by_id(id)
        if not station:
            raise HTTPException(status_code=404, detail="주유소를 찾을 수 없습니다.")
        
        lat = station.get('위도', 0)
        lng = station.get('경도', 0)
        name = station.get('상호', '주유소')
        address = station.get('주소', '')
        
        # 2. 추천 결과
        try:
            recommendations = recommend_service.recommend_by_query(address, top_k=5)
            rec_items = recommendations.get('items', [])
        except:
            rec_items = []
        
        # 3. 지적도 + 지도 생성
        m = folium.Map(location=[lat, lng], zoom_start=17, tiles='OpenStreetMap')
        
        # 3-1. 지적도 오버레이 (필수!)
        parcel_service = get_parcel_service()
        if parcel_service.is_loaded:
            nearby_parcels = parcel_service.get_nearby_parcels(lat, lng, radius=0.003)
            
            if not nearby_parcels.empty:
                # 필지별로 그리기 (최대 200개)
                for idx, row in nearby_parcels.head(200).iterrows():
                    # 면적 계산
                    area = row.geometry.area * (111000 ** 2)
                    
                    # 크기별 색상
                    if area < 300:
                        color = '#3498db'  # 파랑
                        label = '소형'
                    elif area < 1000:
                        color = '#2ecc71'  # 초록
                        label = '중형'
                    elif area < 3000:
                        color = '#f39c12'  # 주황
                        label = '대형'
                    else:
                        color = '#e74c3c'  # 빨강
                        label = '초대형'
                    
                    folium.GeoJson(
                        row.geometry,
                        style_function=lambda x, c=color: {
                            'fillColor': c,
                            'color': 'black',
                            'weight': 0.5,
                            'fillOpacity': 0.4
                        },
                        tooltip=f"{label} - {row.get('JIBUN', 'N/A')} - {area:.0f}㎡"
                    ).add_to(m)
        
        # 3-2. 주유소 마커
        folium.Marker(
            [lat, lng],
            popup=f"<b>{name}</b><br>{address}",
            tooltip=name,
            icon=folium.Icon(color='red', icon='gas-pump', prefix='fa')
        ).add_to(m)
        
        # 3-3. 반경 표시
        folium.Circle(
            [lat, lng],
            radius=300,
            color='red',
            fill=True,
            fillOpacity=0.1,
            popup='반경 300m'
        ).add_to(m)
        
        # 범례 추가
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; 
                    background: white; padding: 15px; border: 2px solid gray; 
                    border-radius: 5px; z-index: 9999;">
            <p style="margin: 0 0 10px 0; font-weight: bold;">필지 크기</p>
            <p style="margin: 5px 0;">
                <span style="background: #3498db; padding: 3px 10px;">　</span> 소형 (&lt;300㎡)
            </p>
            <p style="margin: 5px 0;">
                <span style="background: #2ecc71; padding: 3px 10px;">　</span> 중형 (300-1000㎡)
            </p>
            <p style="margin: 5px 0;">
                <span style="background: #f39c12; padding: 3px 10px;">　</span> 대형 (1000-3000㎡)
            </p>
            <p style="margin: 5px 0;">
                <span style="background: #e74c3c; padding: 3px 10px;">　</span> 초대형 (&gt;3000㎡)
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        map_html = m._repr_html_()
        
        # 4. 추천 결과 HTML
        recommendations_html = ""
        for i, item in enumerate(rec_items[:5], 1):
            recommendations_html += f"""
            <div style="padding: 12px; margin: 8px 0; background: white; 
                        border-left: 4px solid #3498db; border-radius: 3px;">
                <strong>{i}. {item.get('type', '')}</strong>
                <span style="color: #7f8c8d; margin-left: 10px;">
                    점수: {item.get('score', 0):.3f}
                </span>
                <br>
                <small style="color: #34495e;">{item.get('description', '')}</small>
            </div>
            """
        
        # 5. HTML 조합
        html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="utf-8">
            <title>{name} 입지 분석 보고서</title>
            <style>
                body {{ font-family: Arial; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                             border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea, #764ba2); 
                          color: white; padding: 30px; }}
                .section {{ padding: 25px; border-bottom: 1px solid #eee; }}
                .map-container {{ height: 500px; }}
                h1 {{ margin: 0 0 10px 0; }}
                h2 {{ color: #2c3e50; margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📍 {name}</h1>
                    <p>{address}</p>
                </div>
                
                <div class="section">
                    <h2>🗺️ 위치 및 필지 지도</h2>
                    <div class="map-container">{map_html}</div>
                    <p style="margin-top: 10px; color: #7f8c8d; font-size: 13px;">
                        ※ 색상은 필지 크기를 나타냅니다. 
                        빨간 원은 반경 300m 범위입니다.
                    </p>
                </div>
                
                <div class="section">
                    <h2>💡 추천 활용방안</h2>
                    {recommendations_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"보고서 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
        
        df = service.data.get("closed_gas_station")
        print("컬럼:", df.columns.tolist())
        print("id 앞부분:", df.head(5))
        
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