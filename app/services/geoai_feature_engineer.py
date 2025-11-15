# app/services/geoai_feature_engineer.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from app.services.parcel_service import ParcelService
from app.utils.address_utils import extract_sidocode
from app.services.geoai_config import GeoAIConfig

class GeoAIFeatureEngineer:
    def __init__(self):
        self.cfg = GeoAIConfig()
        self.parcels = ParcelService()

    def run(self):
        print("📂 station.csv 로드 중...")
        stations = pd.read_csv(self.cfg.station_csv)

        print("📂 train.csv 로드 중...")
        train = pd.read_csv(self.cfg.train_csv)

        # geodataframe 생성
        gdf_station = gpd.GeoDataFrame(
            stations,
            geometry=gpd.points_from_xy(stations["_X"], stations["_Y"]),
            crs="EPSG:4326"
        )

        gdf_train = gpd.GeoDataFrame(
            train,
            geometry=gpd.points_from_xy(train["경도"], train["위도"]),
            crs="EPSG:4326"
        )

        # sidocode 기반으로 SHP 병합 처리
        features = []
        for idx, row in train.iterrows():
            sidocode = extract_sidocode(row["adm_cd"])
            parcel_gdf = self.parcels.load_parcels(sidocode)

            pt = Point(row["경도"], row["위도"])

            # 300m, 500m buffer
            b300 = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(3857).buffer(300).to_crs(4326)
            b500 = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(3857).buffer(500).to_crs(4326)

            # 필지 카운트
            count_300 = parcel_gdf[parcel_gdf.intersects(b300[0])].shape[0]
            count_500 = parcel_gdf[parcel_gdf.intersects(b500[0])].shape[0]

            features.append({
                "id": idx,
                "parcel_300m": count_300,
                "parcel_500m": count_500
            })

        df_feat = pd.DataFrame(features)
        train = pd.concat([train.reset_index(drop=True), df_feat], axis=1)

        print("✅ GeoAI 필지 기반 피처 생성 완료")
        return train
