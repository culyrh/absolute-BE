# app/services/parcel_service.py

import geopandas as gpd
import os
from app.services.geoai_config import GeoAIConfig
from functools import lru_cache


class ParcelService:
    def __init__(self):
        self.cfg = GeoAIConfig()
        self.cache = {}  # 시도코드 → geodataframe

    def load_parcels(self, sidocode: str) -> gpd.GeoDataFrame:
        if sidocode in self.cache:
            return self.cache[sidocode]

        folder = self.cfg.parcel_base_dir / sidocode
        folder = str(folder)

        # 폴더 내에 shp 한 개만 있다고 가정
        shp_files = [f for f in os.listdir(folder) if f.endswith(".shp")]
        if not shp_files:
            raise FileNotFoundError(f"❌ shp 파일 없음: {folder}")

        shp_path = os.path.join(folder, shp_files[0])
        gdf = gpd.read_file(shp_path)
        gdf = gdf.to_crs(epsg=4326)

        self.cache[sidocode] = gdf
        return gdf

@lru_cache()
def get_parcel_service() -> ParcelService:
    """FastAPI 의존성 주입용 팩토리 함수"""
    return ParcelService()