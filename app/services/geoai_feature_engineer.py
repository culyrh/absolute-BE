# app/services/geoai_feature_engineer.py

import math
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
import os

from app.services.geoai_config import GeoAIConfig


class GeoAIFeatureEngineer:
    """
    GeoAI용 공간 피처 엔지니어링 (축소 버전, 버퍼 안 점 시각화용 좌표 제외)
    ------------------------------------------------------------
    - parcels.geom : EPSG:5186 (Polygon/MultiPolygon)
    - poi.geom     : EPSG:5186 (Point)

    생성되는 피처:
        • parcel_300m        : 300m 이내 필지 개수
        • parcel_500m        : 500m 이내 필지 개수
        • nearest_parcel_m   : 가장 가까운 필지까지의 거리(m)
        • poi_store_300m     : 300m 이내 편의점 개수
        • poi_hotel_300m     : 300m 이내 숙박시설 개수
        • poi_restaurant_300m: 300m 이내 음식점 개수

    👉 버퍼 안 개별 점 좌표(json) 컬럼은 생성하지 않는다.
    """

    def __init__(self, debug: bool = True, debug_limit: int = 5):
        self.cfg = GeoAIConfig()
        self.debug = debug
        self.debug_limit = debug_limit

        if self.debug:
            print("🚀 GeoAI FeatureEngineer (축소버전) 활성화")

        load_dotenv()  # .env 자동 로드
        
        # DB 연결 (필요하면 여기만 수정)
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        self.conn.autocommit = False
        self.cur = self.conn.cursor(cursor_factory=DictCursor)

    # -------------------------------------------------------------------------
    # 핵심: df(Point 목록)에 대해 parcel/poi 피처를 1개의 PostGIS 쿼리로 배치 계산
    # -------------------------------------------------------------------------
    def _compute_all_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        points = []
        for idx, row in df.iterrows():
            try:
                lat = float(row["위도"])
                lon = float(row["경도"])
                points.append((idx, lon, lat))  # id = df.index 사용
            except Exception:
                continue

        if self.debug:
            print(f"📌 배치 계산 대상 포인트 수: {len(points)}")

        if not points:
            # 한 점도 없으면 빈 DF 리턴
            return pd.DataFrame(
                columns=[
                    "parcel_300m", "parcel_500m", "nearest_parcel_m",
                    "poi_store_300m", "poi_hotel_300m", "poi_restaurant_300m",
                ]
            )

        # VALUES 절 생성
        values_sql_parts = []
        params: List[float] = []

        for row_id, lon, lat in points:
            values_sql_parts.append("(%s, %s, %s)")
            params.extend([int(row_id), float(lon), float(lat)])

        values_clause = ",\n        ".join(values_sql_parts)

        # ---------------- PostGIS 배치 SQL (좌표 리스트 없이 집계만) ----------------
        sql = f"""
        WITH pts AS (
            SELECT
                id,
                ST_Transform(
                    ST_SetSRID(ST_Point(lon, lat), 4326),
                    5186
                ) AS geom
            FROM (VALUES
                {values_clause}
            ) AS v(id, lon, lat)
        ),

        -- pts × parcels (500m 이내) 전체 매칭
        parcel_hits AS (
            SELECT
                pts.id AS pid,
                p.geom AS parcel_geom
            FROM pts
            JOIN parcels p
              ON p.geom && ST_Expand(pts.geom, 500)
             AND ST_DWithin(p.geom, pts.geom, 500)
        ),

        -- pts × poi (300m 이내) 전체 매칭
        poi_hits AS (
            SELECT
                pts.id AS pid,
                u.geom AS poi_geom,
                u.category
            FROM pts
            JOIN poi u
              ON u.geom && ST_Expand(pts.geom, 300)
             AND ST_DWithin(u.geom, pts.geom, 300)
        ),

        -- parcel 카운트 + 최근접 거리
        parcel_agg AS (
            SELECT
                pt.id,
                COUNT(ph.parcel_geom) FILTER (
                    WHERE ST_Distance(ph.parcel_geom, pt.geom) <= 300
                ) AS parcel_300m,
                COUNT(ph.parcel_geom) AS parcel_500m,
                MIN(ST_Distance(ph.parcel_geom, pt.geom)) AS nearest_parcel_m
            FROM pts pt
            LEFT JOIN parcel_hits ph ON pt.id = ph.pid
            GROUP BY pt.id
        ),

        -- poi 카테고리별 카운트 (300m)
        poi_agg AS (
            SELECT
                pt.id,
                COUNT(*) FILTER (
                    WHERE ph.category = '편의점'
                ) AS poi_store_300m,
                COUNT(*) FILTER (
                    WHERE ph.category = '숙박시설'
                ) AS poi_hotel_300m,
                COUNT(*) FILTER (
                    WHERE ph.category = '음식점'
                ) AS poi_restaurant_300m
            FROM pts pt
            LEFT JOIN poi_hits ph ON pt.id = ph.pid
            GROUP BY pt.id
        )

        SELECT
            pt.id,
            COALESCE(pa.parcel_300m, 0)        AS parcel_300m,
            COALESCE(pa.parcel_500m, 0)        AS parcel_500m,
            COALESCE(pa.nearest_parcel_m, 0.0) AS nearest_parcel_m,
            COALESCE(po.poi_store_300m, 0)     AS poi_store_300m,
            COALESCE(po.poi_hotel_300m, 0)     AS poi_hotel_300m,
            COALESCE(po.poi_restaurant_300m, 0)AS poi_restaurant_300m
        FROM pts pt
        LEFT JOIN parcel_agg pa ON pt.id = pa.id
        LEFT JOIN poi_agg    po ON pt.id = po.id
        ORDER BY pt.id;
        """

        if self.debug:
            print("🧾 GeoAI 배치 SQL 실행 중...")

        self.cur.execute(sql, params)
        rows = self.cur.fetchall()

        feat_map: Dict[int, Dict] = {}
        for r in rows:
            feat_map[int(r["id"])] = {
                "parcel_300m": r["parcel_300m"] or 0,
                "parcel_500m": r["parcel_500m"] or 0,
                "nearest_parcel_m": float(r["nearest_parcel_m"] or 0.0),
                "poi_store_300m": r["poi_store_300m"] or 0,
                "poi_hotel_300m": r["poi_hotel_300m"] or 0,
                "poi_restaurant_300m": r["poi_restaurant_300m"] or 0,
            }

        # df.index 기준으로 다시 정렬 + 없는 건 0으로 채움
        features = []
        for idx in df.index:
            base = {
                "parcel_300m": 0,
                "parcel_500m": 0,
                "nearest_parcel_m": 0.0,
                "poi_store_300m": 0,
                "poi_hotel_300m": 0,
                "poi_restaurant_300m": 0,
            }
            if idx in feat_map:
                base.update(feat_map[idx])
            features.append(base)

        return pd.DataFrame(features, index=df.index)

    # -------------------------------------------------------------------------
    # train.csv용 FeatureEngineering
    # -------------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        print("📂 train.csv 로드 중...")
        train = pd.read_csv(self.cfg.train_csv)

        required = {"위도", "경도", "대분류"}
        if not required.issubset(train.columns):
            raise ValueError("train.csv에 필요한 컬럼이 없습니다: " + ", ".join(required))

        print("🧮 GeoAI Feature Engineering (train, 배치모드) 시작...")
        df_feat = self._compute_all_features_batch(train)

        result = pd.concat(
            [train.reset_index(drop=True), df_feat.reset_index(drop=True)],
            axis=1
        )

        print("✅ 완료: 공간피처 생성됨")
        print("📊 result shape:", result.shape)
        return result

    # -------------------------------------------------------------------------
    # test_data.csv용 FeatureEngineering
    # -------------------------------------------------------------------------
    def run_test(self, test_csv_path: str) -> pd.DataFrame:
        print(f"📂 test CSV 로드 중 → {test_csv_path}")
        df = pd.read_csv(test_csv_path)

        required = {"위도", "경도"}
        if not required.issubset(df.columns):
            raise ValueError("test_data.csv에 필요한 컬럼이 없습니다: " + ", ".join(required))

        print("🧪 Test FeatureEngineering (배치모드) 시작...")
        feat = self._compute_all_features_batch(df)

        result = pd.concat(
            [df.reset_index(drop=True), feat.reset_index(drop=True)],
            axis=1
        )

        print("✅ Test FeatureEngineering 완료")
        print("📊 test result shape:", result.shape)
        return result

    def __del__(self):
        try:
            if self.cur:
                self.cur.close()
            if self.conn:
                self.conn.close()
        except Exception:
            pass
