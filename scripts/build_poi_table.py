import os
import time
import math
import requests
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# ====== 1. 설정 ======
KAKAO_REST_API_KEY = "23f88060feff03f24c4dc64807d2201c"  # 키 교체

# train / test CSV 경로
TRAIN_CSV = "data/train.csv"
TEST_CSV  = "data/test_data.csv"   # 옵션

# Kakao Local API endpoint
KAKAO_LOCAL_URL = "https://dapi.kakao.com/v2/local/search/category.json"

load_dotenv()  # .env 자동 로드

# PostgreSQL 연결 정보
self.conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)


# category_group_code -> (우리 카테고리명)
CATEGORY_CONFIG = {
    "CS2": "편의점",
    "AD5": "숙박시설",
    "FD6": "음식점",
}


# ====== 2. Kakao API 호출 함수 ======
def kakao_category_search(lon, lat, category_code, radius=500, max_pages=3, size=15):
    """
    Kakao Local category search
    - lon, lat: WGS84 좌표
    - category_code: 'CS2', 'AD5', 'FD6'
    - radius: m (최대 20000, 우리는 500m)
    - max_pages: 최대 몇 페이지까지 볼지 (기본 3페이지 => 최대 45개)
    """
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    results = []

    for page in range(1, max_pages + 1):
        params = {
            "category_group_code": category_code,
            "x": lon,
            "y": lat,
            "radius": radius,
            "page": page,
            "size": size,
            "sort": "distance",
        }

        resp = requests.get(KAKAO_LOCAL_URL, headers=headers, params=params, timeout=5)
        if resp.status_code != 200:
            print(f"⚠️ Kakao API 실패: {resp.status_code}, {resp.text[:200]}")
            break

        data = resp.json()
        docs = data.get("documents", [])
        meta = data.get("meta", {})

        results.extend(docs)

        is_end = meta.get("is_end", True)
        if is_end or not docs:
            break

        # 너무 빠르게 때리지 않도록
        time.sleep(0.15)

    return results


# ====== 3. poi 테이블에 INSERT ======
def insert_poi_rows(conn, rows):
    """
    rows: list of dicts
    dict keys: (station_id, src, category, name, address, lat, lon)
    geom은 INSERT 시 PostGIS 함수로 생성
    """
    if not rows:
        return

    with conn.cursor() as cur:
        values = [
            (
                r["station_id"],
                r["src"],
                r["category"],
                r["name"],
                r["address"],
                r["lat"],
                r["lon"],
            )
            for r in rows
        ]

        sql = """
        INSERT INTO poi (station_id, src, category, name, address, lat, lon, geom)
        VALUES %s
        ON CONFLICT (category, lon, lat) DO NOTHING;
        """

        # geom은 서버쪽에서 ST_Transform으로 생성
        template = """
        (%s, %s, %s, %s, %s, %s, %s,
         ST_Transform(ST_SetSRID(ST_Point(%s, %s), 4326), 5186)
        )
        """

        geom_values = [
            (
                v[0], v[1], v[2], v[3], v[4], v[5], v[6],
                v[6], v[5]   # ← 마지막 두 개는 lon, lat (POINT(x,y))
            )
            for v in values
        ]

        # 🔥 여기서 template를 명시해야 함
        execute_values(cur, sql, geom_values, template=template)

    conn.commit()


# ====== 4. 메인 로직: 주유소 리스트 돌면서 POI 수집 ======
def build_poi_from_csv(csv_path, conn, station_offset=0):
    """
    csv_path의 모든 행(각 주유소)에 대해:
      - 반경 500m 편의점/숙박시설/음식점 Kakao 검색
      - poi 테이블에 upsert
    station_id 는 (station_offset + df.index) 로 부여
    """
    print(f"📂 CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)

    if not {"위도", "경도"}.issubset(df.columns):
        raise ValueError("CSV에 '위도', '경도' 컬럼이 필요함")

    total = len(df)
    print(f"🔍 대상 주유소 수: {total}개")

    for idx, row in df.iterrows():
        lat = float(row["위도"])
        lon = float(row["경도"])
        station_id = station_offset + int(idx)

        all_new_rows = []

        for code, cat_name in CATEGORY_CONFIG.items():
            docs = kakao_category_search(lon, lat, code, radius=500, max_pages=3, size=15)
            for d in docs:
                try:
                    place_name = d.get("place_name", "")
                    address_name = d.get("road_address_name") or d.get("address_name") or ""
                    x = float(d.get("x"))
                    y = float(d.get("y"))
                except Exception:
                    continue

                all_new_rows.append({
                    "station_id": station_id,
                    "src": "kakao",
                    "category": cat_name,
                    "name": place_name,
                    "address": address_name,
                    "lat": y,
                    "lon": x,
                })

        insert_poi_rows(conn, all_new_rows)

        if station_id % 50 == 0:
            print(f"✅ 진행중... {station_id+1}/{station_offset+total} 주유소 완료")

        # 너무 빠르게 연속 호출 방지
        time.sleep(0.1)


def main():
    conn = psycopg2.connect(**PG_CONN_INFO)
    try:
        # 기존 poi 전체 삭제하고 새로 만들고 싶으면 주석 해제
        # with conn.cursor() as cur:
        #     cur.execute("TRUNCATE TABLE poi;")
        # conn.commit()

        # 1) train.csv 기준으로 수집
        build_poi_from_csv(TRAIN_CSV, conn, station_offset=0)

        # 2) test_data.csv까지 이어서 넣고 싶으면 주석 해제
        # build_poi_from_csv(TEST_CSV, conn, station_offset=10000)

        print("🎉 POI 수집 및 저장 완료")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
