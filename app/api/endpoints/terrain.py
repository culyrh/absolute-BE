# app/api/endpoints/terrain.py

import os
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import FileResponse

from app.api.dependencies import get_geo_service
from app.services.geo_service import GeoService
from app.services.terrain_service import TerrainMapService

router = APIRouter(
    prefix="/api/stations",
    tags=["gas_stations_terrain"],
)

PG_DSN = os.getenv("POSTGRES_DSN")  # 예: "dbname=absolute user=postgres password=... host=... port=..."

# Terrain 서비스 초기화
terrain_service = TerrainMapService(pg_dsn=PG_DSN)


@router.get("/{id}/terrain")
async def get_station_terrain(
    id: str = Path(..., description="좌표 기반 고유 ID (예: 35689819_128445642)"),
    service: GeoService = Depends(get_geo_service),
):
    """
    특정 주유소 주변 DEM(Hillshade 느낌) + 지적도 + 300m/500m 버퍼를 합친 PNG 이미지 반환
    """
    # 1. 주유소 정보 얻기
    station = service.get_station_by_id(id)
    if not station:
        raise HTTPException(status_code=404, detail="주유소를 찾을 수 없습니다.")

    lat = station.get("위도")
    lng = station.get("경도")

    if lat is None or lng is None:
        raise HTTPException(status_code=500, detail="주유소 좌표가 없습니다.")

    # 2. DEM 모자이크 생성
    dem_img, bbox_3857 = terrain_service.build_dem_mosaic(
        lng, lat, zoom=14, radius_tiles=1
    )

    # 3. 주변 parcels 쿼리
    parcels_wkb = terrain_service._query_parcels_wkb(
        lng, lat, radius_m=500.0
    )

    # 4. overlay 그리기
    final_img = terrain_service.draw_overlay(
        base_img=dem_img,
        bbox_3857=bbox_3857,
        center_lon=lng,
        center_lat=lat,
        parcels_wkb=parcels_wkb,
        radius_300=300.0,
        radius_500=500.0,
    )

    # 5. PNG 저장 후 반환
    out_dir = "generated_maps"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{id}_terrain.png")
    final_img.save(out_path, format="PNG")

    return FileResponse(out_path, media_type="image/png")
