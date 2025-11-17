# app/services/geoai_feature_engineer.py

import pandas as pd
import geopandas as gpd
import psycopg2
from psycopg2.extras import DictCursor
from tqdm import tqdm
from shapely.geometry import Point

from app.services.geoai_config import GeoAIConfig


class GeoAIFeatureEngineer:
    """
    GeoAI용 공간 피처 엔지니어링
    - train.csv의 각 위치(위도, 경도)에 대해
      PostGIS parcels 테이블을 이용해
      300m / 500m 반경 내 필지 개수를 계산해서
      parcel_300m / parcel_500m 컬럼을 생성한다.
    """

    def __init__(self, debug: bool = False, debug_limit: int = 5):
        self.cfg = GeoAIConfig()
        self.debug = debug
        self.debug_limit = debug_limit

        # 디버그 지도 저장 폴더 (지금은 안 써도 됨. 필요하면 folium 다시 붙이면 됨)
        self.debug_dir = self.cfg.data_dir / "debug_maps"
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"🗺 디버그 지도 출력 경로: {self.debug_dir}")

        # 🔧 여기 DSN은 네 환경에 맞게 수정해줘
        # 예: host=localhost dbname=absolute user=postgres password=비번
        self.conn = psycopg2.connect(
            host="127.0.0.1",
            port=5432,
            dbname="absolute",
            user="postgres",
            password="jdf456852!!"  # <- 진짜 비번으로 바꿔
        )

    # --------------------------------------------------
    # PostGIS에서 parcel_300m / parcel_500m 계산
    # --------------------------------------------------
    def _compute_parcel_features(self, lon: float, lat: float) -> dict:
        """
        주어진 위/경도에 대해
        - 300m 내 필지 개수: parcel_300m
        - 500m 내 필지 개수: parcel_500m
        를 PostGIS parcels 테이블에서 계산
        """

        sql = """
        WITH pt AS (
          SELECT ST_Transform(
                   ST_SetSRID(ST_Point(%(lon)s, %(lat)s), 4326),
                   5186
                 ) AS geom
        ),
        buf300 AS (
          SELECT ST_Buffer(geom, 300) AS geom FROM pt
        ),
        buf500 AS (
          SELECT ST_Buffer(geom, 500) AS geom FROM pt
        )
        SELECT
          COUNT(*) FILTER (WHERE ST_Intersects(p.geom, ST_Transform(buf300.geom, 4326))) AS parcel_300m,
          COUNT(*) FILTER (WHERE ST_Intersects(p.geom, ST_Transform(buf500.geom, 4326))) AS parcel_500m
        FROM parcels p, buf300, buf500;
        """

        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(sql, {"lon": lon, "lat": lat})
            row = cur.fetchone()

        return {
            "parcel_300m": row["parcel_300m"] or 0,
            "parcel_500m": row["parcel_500m"] or 0,
        }

    # --------------------------------------------------
    # 메인 실행 함수: train.csv + PostGIS → feature 붙여서 DataFrame 반환
    # --------------------------------------------------
    def run(self) -> pd.DataFrame:
        print("📂 train.csv 로드 중...")
        train = pd.read_csv(self.cfg.train_csv)

        if not {"위도", "경도"}.issubset(train.columns):
            raise ValueError("train.csv에 '위도', '경도' 컬럼이 필요합니다.")

        features = []
        debug_count = 0

        print("🧮 PostGIS parcels를 이용한 공간 피처 계산 중...")

        for idx, row in tqdm(
            train.iterrows(),
            total=len(train),
            desc="GeoAI Feature Engineering (PostGIS)"
        ):
            lat = float(row["위도"])
            lon = float(row["경도"])

            feat = self._compute_parcel_features(lon=lon, lat=lat)
            feat["id"] = idx  # 나중에 병합용

            features.append(feat)

            # (선택) 디버그용으로 몇 개만 콘솔에 찍어보기
            if self.debug and debug_count < self.debug_limit:
                print(
                    f"[DEBUG] idx={idx}, lat={lat:.6f}, lon={lon:.6f} → "
                    f"parcel_300m={feat['parcel_300m']}, parcel_500m={feat['parcel_500m']}"
                )
                debug_count += 1

        df_feat = pd.DataFrame(features)

        # id 기준으로 train과 feature를 합치기
        train = train.reset_index(drop=True)
        df_feat = df_feat.sort_values("id").reset_index(drop=True)

        result = pd.concat([train, df_feat[["parcel_300m", "parcel_500m"]]], axis=1)

        print("✅ GeoAI 필지 기반 피처 생성 완료 (PostGIS)")
        print("   - 생성 컬럼: parcel_300m, parcel_500m")
        print("   - result shape:", result.shape)

        return result
