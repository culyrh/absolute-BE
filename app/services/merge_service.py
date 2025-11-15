# app/services/merge_service.py

import geopandas as gpd
import os
from app.services.geoai_config import GeoAIConfig

class MergeService:
    def __init__(self):
        self.cfg = GeoAIConfig()
        self.cache = {}  # 시도코드 → geodataframe

    def load_parcels(self, sidocode: str) -> gpd.GeoDataFrame:
        if sidocode in self.cache:
            return self.cache[sidocode]

        folder = str(self.cfg.parcel_base_dir / sidocode)
        shp_files = [f for f in os.listdir(folder) if f.endswith(".shp")]

        if not shp_files:
            raise FileNotFoundError(f"❌ SHP 파일 없음: {folder}")

        gdfs = []
        for shp in shp_files:
            path = os.path.join(folder, shp)
            gdf = gpd.read_file(path)
            gdfs.append(gdf)

        # 모든 SHP를 하나로 병합
        full_gdf = pd.concat(gdfs, ignore_index=True)
        full_gdf = full_gdf.to_crs(epsg=4326)

        self.cache[sidocode] = full_gdf
        return full_gdf

