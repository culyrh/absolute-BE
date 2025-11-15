# app/services/geoai_feature_engineer.py

import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import Point
from pathlib import Path
import folium

from app.services.merge_service import MergeService
from app.utils.address_utils import extract_sidocode
from app.services.geoai_config import GeoAIConfig


# --------------------------------------------------
# VWorld 고해상도 지도 레이어 추가 함수
# --------------------------------------------------
def add_vworld_layers(m, api_key: str):
    # 기본 지도(Base)
    folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Base/{{z}}/{{x}}/{{y}}.png",
        attr="VWorld Base Map",
        name="VWorld Base",
        overlay=False,
        control=True
    ).add_to(m)

    # 위성(항공사진)
    folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Satellite/{{z}}/{{x}}/{{y}}.jpeg",
        attr="VWorld Satellite",
        name="VWorld Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    # 하이브리드(항공 + 라벨)
    folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Hybrid/{{z}}/{{x}}/{{y}}.png",
        attr="VWorld Hybrid",
        name="VWorld Hybrid",
        overlay=True,
        control=True
    ).add_to(m)

    # 지적도 Cadastre (가장 중요한 부분)
    folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{api_key}/Cadastre/{{z}}/{{x}}/{{y}}.png",
        attr="VWorld Cadastre",
        name="VWorld Cadastre (지적도)",
        overlay=True,
        control=True
    ).add_to(m)



# --------------------------------------------------
# GeoAI Feature Engineer
# --------------------------------------------------
class GeoAIFeatureEngineer:
    def __init__(self, debug: bool = False, debug_limit: int = 5):
        """
        debug: True일 때 folium 지도 HTML 생성
        debug_limit: 생성할 디버그 지도 개수
        """
        self.cfg = GeoAIConfig()
        self.parcels = MergeService()
        self.debug = debug
        self.debug_limit = debug_limit

        # HTML 저장 폴더
        self.debug_dir = self.cfg.data_dir / "debug_maps"
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"🗺 디버그 지도 출력 경로: {self.debug_dir}")

        # ★ 반드시 VWorld API 키 넣기
        self.vworld_key = "7233BE3D-BED5-3B78-AE4B-B1D66C2D995C"


    # --------------------------------------------------
    # 지도 디버그 HTML 생성
    # --------------------------------------------------
    def _create_debug_map(self, idx, row, b300, b500, parcels_300, parcels_500, sidocode):
        center_lat = row["위도"]
        center_lng = row["경도"]

        # folium 지도 기본 생성
        m = folium.Map(location=[center_lat, center_lng], zoom_start=17)

        # VWorld 타일 레이어 추가
        add_vworld_layers(m, self.vworld_key)

        # 중심점 표시
        folium.Marker(
            location=[center_lat, center_lng],
            tooltip=f"station idx={idx}, sidocode={sidocode}"
        ).add_to(m)

        # 300m, 500m 버퍼 영역
        folium.GeoJson(b300.__geo_interface__, name="buffer_300m").add_to(m)
        folium.GeoJson(b500.__geo_interface__, name="buffer_500m").add_to(m)

        # 교차 필지 표시
        if not parcels_300.empty:
            folium.GeoJson(
                parcels_300.__geo_interface__,
                name="parcels_within_300m"
            ).add_to(m)

        if not parcels_500.empty:
            folium.GeoJson(
                parcels_500.__geo_interface__,
                name="parcels_within_500m"
            ).add_to(m)

        folium.LayerControl().add_to(m)

        out_path = self.debug_dir / f"station_{idx}_sidocode_{sidocode}.html"
        m.save(str(out_path))
        print(f"✅ 디버그 지도 저장 완료: {out_path}")



    # --------------------------------------------------
    # 메인 GeoAI Feature Engineering 실행
    # --------------------------------------------------
    def run(self):
        print("📂 station.csv 로드 중...")
        stations = pd.read_csv(self.cfg.station_csv)

        print("📂 train.csv 로드 중...")
        train = pd.read_csv(self.cfg.train_csv)

        # geometry 생성 (필요 시 활용)
        gdf_train = gpd.GeoDataFrame(
            train,
            geometry=gpd.points_from_xy(train["경도"], train["위도"]),
            crs="EPSG:4326"
        )

        features = []   # 피처 저장용 리스트
        debug_count = 0

        for idx, row in tqdm(train.iterrows(),      # tqdm으로 진행상황 표시
                             total=len(train),
                            desc="GeoAI Feature Engineering"):
            sidocode = extract_sidocode(row["adm_cd"])      # 시도코드 추출 (32123456 -> 32)
            parcel_gdf = self.parcels.load_parcels(sidocode)

            pt = Point(row["경도"], row["위도"])


            # ----------------------
            # 버퍼 생성
            # ----------------------
            b300 = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(3857).buffer(300).to_crs(4326)[0]
            b500 = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(3857).buffer(500).to_crs(4326)[0]

            # ----------------------
            # 필지 교차
            # ----------------------
            intersect_300 = parcel_gdf[parcel_gdf.intersects(b300)]
            intersect_500 = parcel_gdf[parcel_gdf.intersects(b500)]

            features.append({
                "id": idx,
                "parcel_300m": intersect_300.shape[0],
                "parcel_500m": intersect_500.shape[0]
            })

            # ----------------------
            # 디버그 지도 생성 (상위 debug_limit개)
            # ----------------------
            if self.debug and debug_count < self.debug_limit:
                print(
                    f"🧪 디버그 지도 생성: idx={idx}, sidocode={sidocode}, "
                    f"parcel300={intersect_300.shape[0]}, parcel500={intersect_500.shape[0]}"
                )
                self._create_debug_map(
                    idx, row,
                    b300=b300,
                    b500=b500,
                    parcels_300=intersect_300,
                    parcels_500=intersect_500,
                    sidocode=sidocode
                )
                debug_count += 1

        df_feat = pd.DataFrame(features)
        train = pd.concat([train.reset_index(drop=True), df_feat], axis=1)

        print("✅ GeoAI 필지 기반 피처 생성 완료")
        return train
