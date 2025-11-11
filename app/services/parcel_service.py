"""
필지(지적도) 데이터 처리 서비스
app/services/parcel_service.py
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Optional
from shapely.geometry import Point


class ParcelService:
    """지적도 데이터 관리"""
    
    def __init__(self, parcel_dir: str = "data/parcels"):
        self.parcel_dir = parcel_dir
        self.parcels_gdf = None
        self.is_loaded = False
        
    def load_parcels(self):
        """Shapefile 로딩 (서버 시작 시 1회)"""
        parcel_path = Path(self.parcel_dir)
        if not parcel_path.exists():
            print(f"⚠️ 지적도 디렉토리 없음: {self.parcel_dir}")
            return False
            
        shapefiles = list(parcel_path.glob("**/*.shp"))
        if not shapefiles:
            print(f"⚠️ Shapefile 없음")
            return False
        
        print(f"📂 {len(shapefiles)}개 Shapefile 로딩 중...")
        gdf_list = []
        
        for shp in shapefiles:
            try:
                gdf = gpd.read_file(str(shp))
                if gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                gdf_list.append(gdf)
            except Exception as e:
                print(f"⚠️ {shp.name} 로딩 실패: {e}")
        
        if gdf_list:
            self.parcels_gdf = pd.concat(gdf_list, ignore_index=True)
            self.is_loaded = True
            print(f"✅ 총 {len(self.parcels_gdf)} 필지 로딩 완료")
            return True
        
        return False
    
    def get_nearby_parcels(self, lat: float, lng: float, radius: float = 0.005):
        """주변 필지 가져오기 (radius: 도 단위, 0.005 ≈ 500m)"""
        if not self.is_loaded or self.parcels_gdf is None:
            return gpd.GeoDataFrame()
        
        point = Point(lng, lat)
        buffer = point.buffer(radius)
        
        nearby = self.parcels_gdf[self.parcels_gdf.geometry.intersects(buffer)]
        return nearby


# 전역 인스턴스 (싱글톤)
_parcel_service = None

def get_parcel_service() -> ParcelService:
    """의존성 주입용 함수"""
    global _parcel_service
    if _parcel_service is None:
        _parcel_service = ParcelService()
        _parcel_service.load_parcels()
    return _parcel_service