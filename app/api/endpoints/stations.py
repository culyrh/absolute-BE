"""
주유소 정보 관련 API 엔드포인트
"""

from collections import Counter
from html import escape
from typing import Optional, List, Dict, Any
from datetime import datetime


import json
import traceback
import pandas as pd
import folium
import math
import requests
from fastapi import APIRouter, Depends, Query, HTTPException, Path
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from shapely.geometry import Point

from app.api.dependencies import get_geo_service, get_report_service
from app.schemas.gas_station import GasStationList, GasStationResponse
from app.services.geo_service import GeoService
from app.services.parcel_service import get_parcel_service
from app.services.report_service import LLMReportService
from app.services.terrain_service import TerrainMapService
from app.core.config import DATA_DIR


from dotenv import load_dotenv
load_dotenv()

from app.core.config import get_settings
settings = get_settings()

router = APIRouter(
    prefix="/api/stations",
    tags=["gas_stations"],
    responses={404: {"description": "Not found"}},
)


METERS_PER_DEGREE = 111_000


def _classify_parcel_area(area_m2: float) -> str:
    if area_m2 < 300:
        return "소형"
    if area_m2 < 1000:
        return "중형"
    if area_m2 < 3000:
        return "대형"
    return "초대형"


def _extract_land_use(row: Dict[str, Any]) -> Optional[str]:
    candidate_keys = [
        "JIMOK",
        "JIGU",
        "USEDSGN",
        "USE",
        "LAND_USE",
        "ZONING",
        "지목",
        "용도지역",
    ]
    for key in candidate_keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def _format_recommendations_from_api_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """/{id}/recommend API 응답을 보고서에서 사용하는 포맷으로 변환한다."""

    if not isinstance(payload, dict):
        return []

    formatted: List[Dict[str, Any]] = []
    for rank in range(1, 6):
        key = f"recommend{rank}"
        usage_value = payload.get(key)
        if usage_value is None:
            continue

        usage_text = str(usage_value).strip()
        if not usage_text or usage_text.lower() == "nan":
            continue

        formatted.append(
            {
                "type": usage_text,
                "rank": rank,
                "source": "station_recommend_api",
                "description": f"추천 API 결과 순위 {rank}위",
            }
        )

    return formatted


def _summarise_nearby_parcels(gdf, lat: float, lng: float) -> Optional[Dict[str, Any]]:
    if gdf is None or getattr(gdf, "empty", True):
        return None

    bucket_counter: Counter[str] = Counter()
    total_area = 0.0
    land_use_counter: Counter[str] = Counter()
    closest_info: Optional[Dict[str, Any]] = None
    station_point = Point(lng, lat)

    for _, row in gdf.iterrows():
        geometry = row.get("geometry")
        if geometry is None or geometry.is_empty:
            continue

        try:
            area_m2 = abs(float(geometry.area)) * (METERS_PER_DEGREE ** 2)
        except Exception:
            area_m2 = 0.0

        if area_m2 > 0:
            bucket_counter[_classify_parcel_area(area_m2)] += 1
            total_area += area_m2

        land_use = _extract_land_use(row)
        if land_use:
            land_use_counter[land_use] += 1

        try:
            distance_m = geometry.centroid.distance(station_point) * METERS_PER_DEGREE
        except Exception:
            distance_m = None

        if distance_m is not None:
            if not closest_info or distance_m < closest_info.get("distance_m", float("inf")):
                closest_info = {
                    "distance_m": float(distance_m),
                    "label": row.get("JIBUN") or row.get("PNU") or row.get("LOTNO") or row.get("BUNJI"),
                }

    total_count = sum(bucket_counter.values())
    if total_count == 0:
        return None

    average_area = total_area / total_count if total_count else 0
    top_land_uses = [
        {"use": use, "count": count}
        for use, count in land_use_counter.most_common(3)
    ]

    return {
        "total_count": total_count,
        "total_area": total_area,
        "average_area": average_area,
        "bucket_counts": dict(bucket_counter),
        "top_land_uses": top_land_uses,
        "closest": closest_info,
    }


def kakao_local_search(query: str):
    """
    Kakao Local API — 반경 검색 없이 query 기반 검색
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {settings.KAKAO_REST_API_KEY}"
    }

    params = {
        "query": query,
        "size": 15
    }

    r = requests.get(url, headers=headers, params=params)

    if r.status_code != 200:
        return []

    docs = r.json().get("documents", [])
    results = []

    for d in docs:
        try:
            results.append({
                "name": d.get("place_name"),
                "lat": float(d.get("y")),
                "lng": float(d.get("x")),
                "address": d.get("address_name"),
                "road_address": d.get("road_address_name")
            })
        except:
            continue

    return results


# ============================================================
# 필지(개별공시지가 + 토지이용계획) 정보 CSV 로더
# ============================================================

LAND_PRICE_PATH = DATA_DIR / "station_with_landprice.csv"
LAND_USE_PATH = DATA_DIR / "station_with_landuse.csv"

# CSV는 lazy loading으로만 처리 → import 시점에 절대 읽지 않음
def load_land_price_df():
    try:
        return pd.read_csv(LAND_PRICE_PATH, dtype=str)
    except Exception as e:
        print("⚠ land price csv 로딩 오류:", e)
        return None

def load_land_use_df():
    try:
        return pd.read_csv(LAND_USE_PATH, dtype=str)
    except Exception as e:
        print("⚠ land use csv 로딩 오류:", e)
        return None

def _classify_landuse(code: str, name: str) -> str:
    """
    용도지역지구코드 → 대분류 카테고리 분류
    - 'zoning'       : 용도지역/지구 (주거·상업·공업·녹지, 관리지역 등)
    - 'infra'        : 도로·철도·주차장·하천 등 기반시설
    - 'environment'  : 환경·보전·규제 권역
    - 'development'  : 개발행위·지구단위·도시관리계획 등
    - 'other'        : 위에 안 걸리는 나머지
    """
    code = (code or "").upper()

    # 용도지역·용도지구
    if code.startswith(("UQA", "UQB", "UQC", "UQD")):
        return "zoning"

    # 도로, 철도, 주차장, 하천 등 기반시설
    if code.startswith(("UIA", "UIK", "UQS", "UJB", "UQW")) or code in {
        "UQS200", "UQS210", "UQS510"
    }:
        return "infra"

    # 환경·보전·규제 권역
    if code.startswith(("UMZ", "UMN", "UMX", "UG", "UOC")) or code in {
        "UBA100", "UBA200", "UBA300", "UDV100"
    }:
        return "environment"

    # 개발 관련(지구단위, 개발제한구역, 성장/개발지구 등)
    if code.startswith(("UQQ", "UQN", "UQM", "UHA", "UHG", "UHJ", "UM2", "UFM", "UHB", "UHD")):
        return "development"

    return "other"



# ============================================================
# API 엔드포인트
# ============================================================

@router.get("/region/{code:path}")
async def get_geojson_by_region(
    code: str = Path(..., description="지역 코드 (예: 서울특별시, 전주시 등)"),
    limit: int = Query(5000, ge=1, le=5000, description="반환할 결과 수"),
    service: GeoService = Depends(get_geo_service),
):
    """
    지역별 주유소 목록 GeoJSON API
    """
    try:
        # 지역 데이터 조회
        result = service.search_by_address(code, limit)
        if not result:
            return JSONResponse(content={"type": "FeatureCollection", "features": []})

        # GeoJSON 형태로 변환
        features = []
        for item in result:
            try:
                lon = float(item.get("경도"))
                lat = float(item.get("위도"))
            except (ValueError, TypeError):
                continue  # 좌표 없는 항목은 제외

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    k: v for k, v in item.items()
                    if k not in ["경도", "위도"]
                }
            }
            features.append(feature)

        # GeoJSON 반환
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        headers = {"Cache-Control": "public, max-age=3600"}
        return JSONResponse(content=geojson, headers=headers)

    except Exception as e:
        print(f"지역별 GeoJSON 변환 오류: {e}")
        raise HTTPException(status_code=500, detail=f"GeoJSON 변환 중 오류 발생: {e}")


@router.get("/map", response_model=GasStationList)
async def get_stations_in_map(
    lat1: float = Query(..., description="위도 최소값"),
    lng1: float = Query(..., description="경도 최소값"),
    lat2: float = Query(..., description="위도 최대값"),
    lng2: float = Query(..., description="경도 최대값"),
    limit: int = Query(10000, ge=1, le=10000, description="반환할 결과 수"),
    service: GeoService = Depends(get_geo_service),
):
    """
    지도 범위 내 주유소 API
    
    - **lat1**: 위도 최소값 (필수)
    - **lng1**: 경도 최소값 (필수)
    - **lat2**: 위도 최대값 (필수)
    - **lng2**: 경도 최대값 (필수)
    - **limit**: 반환할 결과 수 (기본값: 10000, 최대: 10000)
    """
    try:
        # 폐휴업 주유소 데이터에서 좌표로 검색

        # preprocess_gas_station_data의 processed_df 반환 
        # -> (행정구역, 권역) 컬럼 추가 / idx가 부여된 station 데이터
        gas_df = service.data.get("gas_station", None)
        
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
        
        filtered_df = filtered_df[
            filtered_df["위도"].apply(lambda x: isinstance(x, (int, float))) &
            filtered_df["경도"].apply(lambda x: isinstance(x, (int, float)))
        ]

        # NaN → None 변환
        clean_df = filtered_df.where(filtered_df.notnull(), None)

        # 결과 형식화
        result = clean_df.head(limit).to_dict("records")

        # JSON 직렬화 오류 해결 / 모든 속성의 결측치 제거
        def sanitize_value(v):
            if v is None:
                return None
            # NaN 또는 Infinite → None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        # 모든 레코드에 대해 NaN/inf 정리
        result = [
            {k: sanitize_value(v) for k, v in item.items()}
            for item in result
        ]            
        
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
    query: str = Query(..., description="주유소 이름 검색어"),
    limit: int = Query(100, ge=1, le=1000, description="반환할 결과 수"),
    service: GeoService = Depends(get_geo_service),
):
    """
    주유소명 기반 검색 API

    - **query**: 주유소명 검색어 (예: '현대', 'SK', '목화')
    - **limit**: 반환할 결과 수 (기본값: 100, 최대: 1000)
    """
    try:
        # 주유소 이름으로 검색
        result = service.search_by_name(query, limit)
        
        # GeoJSON 형식으로 반환
        features = []
        for item in result:
            try:
                lon = float(item.get("경도"))
                lat = float(item.get("위도"))
            except (ValueError, TypeError):
                continue

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    k: v for k, v in item.items() if k not in ["경도", "위도"]
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        return JSONResponse(content=geojson)

    except Exception as e:
        print(f"주유소명 기반 검색 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"주유소명 기반 검색 중 오류 발생: {str(e)}")



# ============================================================
# 주유소 개별 정보 API
# ============================================================

BJD_PATH = DATA_DIR / "법정동_코드_전체자료.csv"
BJD_DF = None

def load_bjd_df():
    global BJD_DF
    if BJD_DF is not None:
        return BJD_DF

    df = pd.read_csv(BJD_PATH, dtype=str)

    def norm(code):
        s = str(code).strip()
        if s.endswith(".0"):
            s = s[:-2]
        s = "".join(c for c in s if c.isdigit())
        if len(s) == 8:
            s += "00"
        if len(s) < 10:
            s = s.ljust(10, "0")
        return s[:10]

    df["법정동코드"] = df["법정동코드"].apply(norm)
    BJD_DF = df
    return df

def get_bjd_name_from_adm(adm_cd2):
    """adm_cd2 → 정규화 → 법정동명 반환"""
    if adm_cd2 is None:
        return None

    df = load_bjd_df()

    s = str(adm_cd2).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(c for c in s if c.isdigit())
    if len(s) == 8:
        s += "00"
    if len(s) < 10:
        s = s.ljust(10, "0")
    s = s[:10]

    row = df[df["법정동코드"] == s]
    if len(row) == 0:
        return None

    return row["법정동명"].iloc[0]

@router.get("/{id}/vehicle")
async def get_vehicle_services(
    id: str = Path(..., description="좌표 기반 고유 ID"),
    service: GeoService = Depends(get_geo_service)
):
    df = service.data.get("gas_station")
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="station.csv 없음")

    # 1) ID → 좌표 복구
    try:
        lat_part, lng_part = id.split("_")
        lat = float(lat_part) / 1_000_000
        lng = float(lng_part) / 1_000_000
    except:
        raise HTTPException(status_code=400, detail="ID 형식 오류")

    # 2) 가장 가까운 station
    df["distance"] = (df["위도"] - lat)**2 + (df["경도"] - lng)**2
    station = df.loc[df["distance"].idxmin()].to_dict()

    # 3) adm_cd2 기반 법정동명 찾기 **(핵심 패치)**
    adm_raw = (
        station.get("법정동코드") or
        station.get("adm_cd2") or
        station.get("법정동 코드")
    )
    region = get_bjd_name_from_adm(adm_raw)

    if not region:
        return {
            "id": id,
            "region": None,
            "정비소": [],
            "세차장": [],
            "타이어": [],
            "카센터": [],
            "total_count": 0
        }

    # 4) Kakao 검색
    repair = kakao_local_search(f"정비소 {region}")
    wash   = kakao_local_search(f"세차장 {region}")
    tire   = kakao_local_search(f"타이어 {region}")
    center = kakao_local_search(f"카센터 {region}")

    total = len(repair) + len(wash) + len(tire) + len(center)

    return {
        "id": id,
        "region": region,
        "정비소": repair,
        "세차장": wash,
        "타이어": tire,
        "카센터": center,
        "total_count": total
    }


# ============================================================

@router.get("/{id}/ev")
async def get_ev_chargers(
    id: str = Path(..., description="좌표 기반 고유 ID"),
    service: GeoService = Depends(get_geo_service)
):
    df = service.data.get("gas_station")
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="station.csv 없음")

    # ID → 좌표 복구
    try:
        lat_part, lng_part = id.split("_")
        lat = float(lat_part) / 1_000_000
        lng = float(lng_part) / 1_000_000
    except:
        raise HTTPException(status_code=400, detail="ID 형식 오류")

    # 가장 가까운 station
    df["distance"] = (df["위도"] - lat)**2 + (df["경도"] - lng)**2
    station = df.loc[df["distance"].idxmin()].to_dict()

    # adm_cd2 기반 법정동명 찾기 **(핵심 패치)**
    adm_raw = (
        station.get("법정동코드") or
        station.get("adm_cd2") or
        station.get("법정동 코드")
    )
    region = get_bjd_name_from_adm(adm_raw)

    if not region:
        return {"id": id, "region": None, "items": [], "count": 0}

    # Kakao 검색
    ev = kakao_local_search(f"전기차충전소 {region}") or []

    return {
        "id": id,
        "region": region,
        "items": ev,
        "count": len(ev)
    }


@router.get("/{id}/recommend")
async def get_station_recommend(
    id: str = Path(..., description="좌표 기반 고유 ID (예: 35689819_128445642)"),
    service: GeoService = Depends(get_geo_service),
):
    """
    좌표 기반 고유 ID로 추천 활용방안 조회
    """
    try:
        df = service.data.get("gas_station")

        # id = "37384645_126941288" → lat,lng 복원
        try:
            lat_part, lng_part = id.split("_")
            lat = float(lat_part) / 1000000
            lng = float(lng_part) / 1000000
        except:
            raise HTTPException(status_code=400, detail="ID 형식 오류")

        # 가까운 station 찾기
        df["distance"] = ((df["위도"] - lat)**2 + (df["경도"] - lng)**2)
        station = df.loc[df["distance"].idxmin()].to_dict()
        station.pop("distance", None)

        return JSONResponse(
            content={
                "id": id,
                "name": station.get("상호"),
                "address": station.get("주소"),
                "recommend1": station.get("recommend1"),
                "recommend2": station.get("recommend2"),
                "recommend3": station.get("recommend3"),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 조회 중 오류: {e}")


@router.get("/{id}/stats")
async def get_station_stats(
    id: str = Path(..., description="좌표 기반 고유 ID (예: 35689819_128445642)"),
    service: GeoService = Depends(get_geo_service),
):
    """
    특정 주유소(id)의 정량 지표 + 권역(train 기반) 비교 API
    - parcel_300m, parcel_500m, 교통량, 관광지수, 인구, 상권밀집도
    - train.csv 기반 시도(region_code)별 평균과 비교
    """

    try:
        # -------------------------------------------
        # 1) station.csv 로딩
        # -------------------------------------------
        df_station = service.data.get("gas_station")
        if df_station is None or df_station.empty:
            raise HTTPException(status_code=500, detail="station.csv 없음")

        df_station = df_station.loc[:, ~df_station.columns.duplicated()]

        # -------------------------------------------
        # 2) 좌표 기반 ID 파싱
        # -------------------------------------------
        try:
            lat_part, lng_part = id.split("_")
            lat = float(lat_part) / 1_000_000
            lng = float(lng_part) / 1_000_000
        except:
            raise HTTPException(status_code=400, detail="ID 형식 오류")

        # -------------------------------------------
        # 3) 가장 가까운 station 찾기
        # -------------------------------------------
        df_station["distance"] = (
            (df_station["위도"] - lat)**2 +
            (df_station["경도"] - lng)**2
        )
        station = df_station.loc[df_station["distance"].idxmin()].to_dict()
        station.pop("distance", None)

        # -------------------------------------------
        # 4-A) station에서 adm_cd2 원본 추출
        # -------------------------------------------
        adm_raw = None

        for key in ["adm_cd2", "법정동코드", "법정동 코드"]:
            if station.get(key) is not None:
                adm_raw = station.get(key)
                break

        # 수정: adm_cd2 없으면 에러 내지 말고 빈 값 반환
        if adm_raw is None or str(adm_raw).strip() == "" or str(adm_raw).lower() == "nan":
            return JSONResponse(
                content={
                    "id": id,
                    "region_code": None,
                    "metrics": {},
                    "train_mean": {},
                    "relative": {},
                    "percentile": {}
                }
            )

        # -------------------------------------------
        # 4-B) adm_cd2 정규화 함수
        # -------------------------------------------
        def normalize_adm_cd2(value):
            if value is None:
                return None

            s = str(value).strip()

            # float 형태 ".0" 제거
            if s.endswith(".0"):
                s = s[:-2]

            # 숫자만 남기기
            s = "".join(ch for ch in s if ch.isdigit())

            # 8자리 법정동 → 10자리 변환
            if len(s) == 8:
                s += "00"

            # 길이 부족하면 0으로 패딩
            if len(s) < 10:
                s = s.ljust(10, "0")

            # 10자리로 자르기
            return s[:10]

        # -------------------------------------------
        # 5) station region_code 생성
        # -------------------------------------------
        adm_cd = normalize_adm_cd2(adm_raw)
        if not adm_cd:
            raise HTTPException(status_code=500, detail="station adm_cd2 오류")

        region_code = adm_cd[:2]

        # -------------------------------------------
        # 6) train.csv 로드
        # -------------------------------------------
        from app.services.geoai_config import GeoAIConfig
        cfg = GeoAIConfig()

        train_path = cfg.data_dir / "train.csv"
        if not train_path.exists():
            raise HTTPException(status_code=500, detail="train.csv 없음")

        df_train = pd.read_csv(train_path)

        df_train["adm_cd2_norm"] = df_train["adm_cd2"].apply(normalize_adm_cd2)
        df_train["region_code"] = df_train["adm_cd2_norm"].str[:2]

        region_train = df_train[df_train["region_code"] == region_code]
        if region_train.empty:
            raise HTTPException(
                status_code=404,
                detail=f"train.csv 에 region_code={region_code} 데이터 없음"
            )

        # -------------------------------------------
        # 7) station ↔ train 지표 매칭
        # -------------------------------------------
        FEATURE_COLS = {
            "traffic": ("교통량", "교통량(AADT)"),
            "tourism": ("관광지수", "숙박업소(관광지수)"),
            "population": ("인구", "인구[명]"),
            "commercial_density": ("상권밀집도", "상권밀집도(비율)"),
            "parcel_300m": ("parcel_300m", "parcel_300m"),
            "parcel_500m": ("parcel_500m", "parcel_500m"),
        }

        # -------------------------------------------
        # 8) station 지표 읽기
        # -------------------------------------------
        metrics = {
            name: station.get(st_col)
            for name, (st_col, tr_col) in FEATURE_COLS.items()
        }

        # -------------------------------------------
        # 9) train 평균 계산
        # -------------------------------------------
        train_mean = {
            name: float(region_train[tr_col].mean())
            for name, (st_col, tr_col) in FEATURE_COLS.items()
            if tr_col in region_train.columns   # 컬럼 존재 확인
        }

        # -------------------------------------------
        # 10) 변화율 계산
        # -------------------------------------------
        def percent_change(a, b):
            if a is None or b is None or b == 0:
                return None
            return (float(a) - float(b)) / float(b) * 100

        relative = {
            name: percent_change(metrics[name], train_mean[name])
            for name in FEATURE_COLS.keys()
            if name in train_mean   # train_mean에 존재하는 지표만
        }

        # -------------------------------------------
        # 11) 백분위 계산
        # -------------------------------------------
        def percentile(series, value):
            if value is None:
                return None
            # 문자열 → 숫자 변환 (오류 방지)
            try:
                value = float(value)
            except:
                return None

            arr = pd.to_numeric(series, errors="coerce").dropna().values
            if len(arr) == 0:
                return None
            
            return float((arr < value).mean() * 100)

        percentiles = {
            name: percentile(region_train[tr_col], metrics[name])
            for name, (st_col, tr_col) in FEATURE_COLS.items()
            if name in train_mean   # train_mean에 존재하는 지표만
        }

        # -------------------------------------------
        # 12) 최종 응답
        # -------------------------------------------
        return JSONResponse(
            content={
                "id": id,
                "region_code": region_code,
                "metrics": metrics,
                "train_mean": train_mean,
                "relative": relative,
                "percentile": percentiles,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{id}/report", response_class=HTMLResponse)
async def generate_station_report(
    id: str = Path(..., description="좌표 기반 고유 ID (예: 35689819_128445642)"),
    service: GeoService = Depends(get_geo_service),
    report_service: LLMReportService = Depends(get_report_service)
):
    """
    주유소 입지 분석 보고서 (지적도 포함)
    - 좌표 기반 고유 ID 사용
    """
    try:
        df = service.data.get("gas_station")

        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="주유소 데이터가 비어있습니다.")

        # ----------------------------------
        # 1) 좌표 기반 ID 파싱
        # ----------------------------------
        try:
            lat_part, lng_part = id.split("_")
            lat = float(lat_part) / 1_000_000
            lng = float(lng_part) / 1_000_000
        except:
            raise HTTPException(status_code=400, detail="ID 형식 오류 (예: 35689819_128445642)")

        # ----------------------------------
        # 2) 가장 가까운 station 찾기
        # ----------------------------------
        df = df.loc[:, ~df.columns.duplicated()]  # 중복된 위도/경도 정리

        df["distance"] = ((df["위도"] - lat)**2 + (df["경도"] - lng)**2)
        nearest_idx = df["distance"].idxmin()
        station = df.loc[nearest_idx].to_dict()
        station.pop("distance", None)

        # station 고유 id는 좌표 id로 재정의
        station_id = id  

        # ----------------------------------
        # 기존 로직 그대로
        # ----------------------------------

        name = station.get('상호', '주유소')
        address = station.get('주소', '')

        # 2. 추천 결과 (/{id}/recommend API 활용)
        combined_recommendations: List[Dict[str, Any]] = []
        try:
            recommend_response = await get_station_recommend(id=id, service=service)
            raw_body = getattr(recommend_response, "body", b"")
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            combined_recommendations = _format_recommendations_from_api_payload(payload)
        except Exception as rec_error:
            print(f"추천 API 호출 오류: {rec_error}")

        parcel_summary = None
        land_payload: Dict[str, Any] = {}

        # 3. 지도 생성
        m = folium.Map(location=[lat, lng], zoom_start=17, tiles='OpenStreetMap')

        try:
            parcel_service = get_parcel_service()
            nearby_parcels = parcel_service.get_nearby_parcels(lat, lng, radius=0.003)
            parcel_summary = _summarise_nearby_parcels(nearby_parcels, lat, lng)
        except Exception as parcel_error:
            print(f"지적도 서비스 오류: {parcel_error}")
            nearby_parcels = None

        try:
            land_response = await get_station_land(id=id, service=service)
            raw_land_body = getattr(land_response, "body", b"")
            land_payload = json.loads(raw_land_body.decode("utf-8")) if raw_land_body else {}
        except Exception as land_error:
            print(f"필지 정보 조회 오류: {land_error}")

        terrain_png_path = f"/api/stations/{station_id}/terrain"

        terrain_img_html = f"""
            <img src="{terrain_png_path}"
                style="width:100%; border-radius:12px; border:1px solid #ccc;">
        """

        map_images = report_service.prepare_map_images(lat, lng)

        stats_payload: Dict[str, Any] = {}
        try:
            stats_response = await get_station_stats(id=id, service=service)
            raw_stats_body = getattr(stats_response, "body", b"")
            stats_payload = json.loads(raw_stats_body.decode("utf-8")) if raw_stats_body else {}
        except Exception as stats_error:
            print(f"분석 지표 조회 오류: {stats_error}")

        llm_report = await report_service.generate_report(
            station,
            combined_recommendations,
            parcel_summary=parcel_summary,
            station_id=station_id,
            map_images=map_images,
            stats_payload=stats_payload,
        )

        # ------------------------------
        # 4. 카카오 지도 (정적처럼 고정)
        # ------------------------------
        js_key = "ef6066b015b62cbc9689dbf67268deb1"

        map_html = f"""
        <div id="report-map"
            style="width:100%;height:100%;border-radius:12px;overflow:hidden;"></div>

        <script>
            (function () {{
            var script = document.createElement('script');
            script.src = "https://dapi.kakao.com/v2/maps/sdk.js?autoload=false&appkey={js_key}&libraries=services";
            script.onload = function () {{
                kakao.maps.load(function () {{
                var container = document.getElementById('report-map');
                if (!container) return;

                // 화면 기본 중심
                var center = new kakao.maps.LatLng({lat}, {lng});
                // 🔽 인쇄할 때는 지도를 조금 위로 올려서(위도 +) 마커가 아래쪽에 보이게
                var printCenter = new kakao.maps.LatLng({lat} + 0.0016, {lng});

                var map = new kakao.maps.Map(container, {{
                    center: center,
                    level: 4
                }});

                // 커스텀 마커
                var imageSrc = "https://absolute-beryl.vercel.app/public/marker_green.png";
                var imageSize = new kakao.maps.Size(36, 43);
                var imageOption = {{ offset: new kakao.maps.Point(18, 43) }};
                var markerImage = new kakao.maps.MarkerImage(imageSrc, imageSize, imageOption);

                var marker = new kakao.maps.Marker({{
                    position: center,
                    image: markerImage
                }});
                marker.setMap(map);

                map.setDraggable(false);
                map.setZoomable(false);
                map.setKeyboardShortcuts(false);

                // 전역에 저장해 두기
                window.reportMap = map;
                window.reportMapCenter = center;
                window.reportMapPrintCenter = printCenter;

                function handleBeforePrint() {{
                    if (!window.reportMap) return;
                    window.reportMap.relayout();
                    window.reportMap.setCenter(window.reportMapPrintCenter);
                }}

                function handleAfterPrint() {{
                    if (!window.reportMap) return;
                    window.reportMap.relayout();
                    window.reportMap.setCenter(window.reportMapCenter);
                }}

                window.addEventListener('beforeprint', handleBeforePrint);
                window.addEventListener('afterprint', handleAfterPrint);
                }});
            }};
            document.head.appendChild(script);
            }})();


        </script>
        """
        # ------------------------------
        html = report_service.build_report_html(
            station=station,
            report_date=datetime.now(),
            map_html=map_html,
            terrain_html=terrain_img_html,
            llm_report=llm_report,
            recommendations=combined_recommendations,
            stats_payload=stats_payload,
            parcel_summary=parcel_summary,
            land_payload=land_payload,
            nearby_parcels_available=nearby_parcels is not None and not nearby_parcels.empty,
            map_images=map_images,
        )

        return HTMLResponse(content=html)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"보고서 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{id}/admin")
async def get_station_admin_info(
    id: str = Path(..., description="좌표 기반 고유 ID"),
    service: GeoService = Depends(get_geo_service),
):
    """
    특정 주유소(id)의 행정동 기준 통계(인구·교통량·상권밀집도·관광지수)
    - station.csv 기반 raw 값을 그대로 사용
    - 행정동 이름은 법정동코드(=adm_cd2) → 법정동_코드.csv 매핑
    """

    # 1) station.csv
    df = service.data.get("gas_station")
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="station.csv 없음")

    df = df.loc[:, ~df.columns.duplicated()]  # 중복 컬럼 제거

    # 2) ID → lat/lng 복구
    try:
        lat_part, lng_part = id.split("_")
        lat = float(lat_part) / 1_000_000
        lng = float(lng_part) / 1_000_000
    except:
        raise HTTPException(status_code=400, detail="ID 형식 오류")

    # 3) 가장 가까운 station 찾기
    df["distance"] = (df["위도"] - lat)**2 + (df["경도"] - lng)**2
    station = df.loc[df["distance"].idxmin()].to_dict()
    station.pop("distance", None)

    # 4) adm_cd2 / 법정동코드 추출
    adm_raw = (
        station.get("법정동코드") or
        station.get("adm_cd2") or
        station.get("법정동 코드")
    )
    if not adm_raw:
        return {
            "id": id,
            "region": None,
            "population": None,
            "traffic": None,
            "commercial_density": None,
            "tourism": None,
        }

    # → 이미 상단에서 로딩한 변환 함수 사용
    region_name = get_bjd_name_from_adm(adm_raw)

    # 5) station 원본 지표 추출
    metrics = {
        "population": station.get("인구") or station.get("인구수"),
        "traffic": station.get("교통량") or station.get("AADT"),
        "commercial_density": station.get("상권밀집도"),
        "tourism": station.get("관광지수"),
    }

    return {
        "id": id,
        "region": region_name,
        **metrics
    }



@router.get("/{id}/land")
async def get_station_land(
    id: str = Path(..., description="좌표 기반 고유 ID (예: 35689819_128445642)"),
    service: GeoService = Depends(get_geo_service),
):
    """
    특정 주유소(id)의 필지 정보 API
    - station_with_landprice.csv + station_with_landuse.csv 기반
    - ID → (lat, lng) 복원 → station.csv에서 가장 가까운 행 찾고, 해당 PNU로 필지 정보 조회
    """
    try:
        # 1) 기본 station.csv 로딩
        df_station = service.data.get("gas_station")
        if df_station is None or df_station.empty:
            raise HTTPException(status_code=500, detail="station.csv 없음")

        df_station = df_station.loc[:, ~df_station.columns.duplicated()]

        if "위도" not in df_station.columns or "경도" not in df_station.columns:
            raise HTTPException(status_code=500, detail="위도/경도 컬럼이 누락되었습니다.")

        if "PNU" not in df_station.columns:
            raise HTTPException(status_code=500, detail="PNU 컬럼이 station.csv 에 없습니다.")

        # 2) 좌표 기반 ID 파싱
        try:
            lat_part, lng_part = id.split("_")
            lat = float(lat_part) / 1_000_000
            lng = float(lng_part) / 1_000_000
        except Exception:
            raise HTTPException(status_code=400, detail="ID 형식 오류")

        # 3) 가장 가까운 station 찾기
        df_station["distance"] = (
            (df_station["위도"] - lat) ** 2
            + (df_station["경도"] - lng) ** 2
        )
        station_row = df_station.loc[df_station["distance"].idxmin()].to_dict()
        station_row.pop("distance", None)

        pnu = str(station_row.get("PNU", "")).strip()
        if not pnu:
            raise HTTPException(status_code=500, detail="_PNU 값이 비어 있습니다.")

        # --------------------------------------------------
        # 4) 개별공시지가 조회 (station_with_landprice.csv)
        # --------------------------------------------------
        df_price = load_land_price_df()
        land_price = None

        if df_price is not None:
            matched_price = df_price[df_price["_PNU"] == pnu]

            if not matched_price.empty:
                matched_price = matched_price.copy()
                # 데이터기준일자 기준 최신 1건 사용
                try:
                    matched_price["데이터기준일자_norm"] = pd.to_datetime(
                        matched_price["데이터기준일자"], errors="coerce"
                    )
                    matched_price = matched_price.sort_values(
                        "데이터기준일자_norm", ascending=False
                    )
                except Exception:
                    pass

                row_p = matched_price.iloc[0]

                raw_price = row_p.get("공시지가", "")
                # 숫자화 (없으면 0)
                try:
                    price_num = int(float(raw_price or 0))
                except Exception:
                    price_num = 0

                land_price = {
                    "price": price_num,
                    "price_str": f"{raw_price}원/㎡" if raw_price != "" else None,
                    "announce_date": str(row_p.get("공시일자", "")),
                    "type": (str(row_p.get("특수지구분명", "")) or None),
                    "data_date": str(row_p.get("데이터기준일자", "")),
                }

        # --------------------------------------------------
        # 5) 토지이용계획(용도지역·지구 등) 조회
        #    station_with_landuse.csv
        # --------------------------------------------------
        df_use = load_land_use_df()
        land_use_summary: Dict[str, List[Dict[str, str]]] = {
            "zoning": [],
            "infra": [],
            "environment": [],
            "development": [],
            "other": [],
        }
        land_use_raw: List[Dict[str, str]] = []

        if df_use is not None:
            matched_use = df_use[df_use["_PNU"] == pnu]

            if not matched_use.empty:
                matched_use = matched_use.copy()
                # 중복 제거 (code + name + date 기준)
                seen_pairs = set()

                for _, row_u in matched_use.iterrows():
                    code = str(row_u.get("용도지역지구코드", "")).strip()
                    name = str(row_u.get("용도지역지구명", "")).strip()
                    base_date = str(row_u.get("데이터기준일자", "")).strip()

                    if not code and not name:
                        continue

                    key = (code, name, base_date)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    cat = _classify_landuse(code, name)
                    item = {
                        "code": code,
                        "name": name,
                        "data_date": base_date,
                    }
                    land_use_summary.setdefault(cat, []).append(item)
                    land_use_raw.append(item)

        # 빈 카테고리는 제거 (프론트에서 다루기 편하게)
        land_use_summary = {k: v for k, v in land_use_summary.items() if v}

        # --------------------------------------------------
        # 6) 최종 응답 구성
        # --------------------------------------------------
        response = {
            "id": id,
            "name": station_row.get("field5")
                    or station_row.get("상호")
                    or station_row.get("name"),
            "address": station_row.get("field6")
                       or station_row.get("주소")
                       or station_row.get("address"),
            "clean_address": station_row.get("_CLEANADDR"),
            "pnu": pnu,
            "location": {
                "lat": float(station_row.get("위도", 0) or 0),
                "lng": float(station_row.get("경도", 0) or 0),
            },
            "land_price": land_price,
            "land_use": {
                "summary": land_use_summary,
                "raw": land_use_raw,
            },
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"필지 정보 조회 오류: {str(e)}")


# ============================================================
# 주유소 지형도 API
# ============================================================

pg_dsn = settings.POSTGRES_DSN
terrain_service = TerrainMapService(pg_dsn)

@router.get("/{id}/terrain")
async def get_station_terrain(
    id: str = Path(...),
    service: GeoService = Depends(get_geo_service),
):
    df = service.data.get("gas_station")
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="station.csv 없음")

    # -----------------------------------------
    # 1) 좌표 기반 ID → 위도/경도 복원
    # -----------------------------------------
    try:
        lat_part, lng_part = id.split("_")
        lat = float(lat_part) / 1_000_000
        lng = float(lng_part) / 1_000_000
    except:
        raise HTTPException(status_code=400, detail="ID 형식 오류")

    # -----------------------------------------
    # 2) 가장 가까운 station 찾기 (report / stats 방식 동일)
    # -----------------------------------------
    df = df.loc[:, ~df.columns.duplicated()]
    df["distance"] = (df["위도"] - lat)**2 + (df["경도"] - lng)**2
    station = df.loc[df["distance"].idxmin()].to_dict()
    station.pop("distance", None)

    # -----------------------------------------
    # 3) terrain 처리
    # -----------------------------------------
    lon = station["경도"]
    lat = station["위도"]

    bbox = terrain_service.compute_bbox_around(lon, lat, meter=500)
    base_img = terrain_service.fetch_hillshade(bbox, width=768, height=768)
    parcels = terrain_service.query_parcels(lon, lat, radius=500)
    final_img = terrain_service.draw_overlay(base_img, bbox, lon, lat, parcels)

    out_dir = "generated_maps"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{id}_terrain.png")
    final_img.save(out_path)

    return FileResponse(out_path, media_type="image/png")


from fastapi.responses import HTMLResponse
from app.services.terrain_service import TerrainMapService

@router.get("/{id}/terrain/html", response_class=HTMLResponse)
async def get_station_terrain_html(
    id: str = Path(...),
    service: GeoService = Depends(get_geo_service),
):
    """
    주유소 주변 300m / 500m 필지 + 지목/용도지역 인터랙티브 지도 (HTML)
    """

    df = service.data.get("gas_station")
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="station.csv 없음")

    # 1) 좌표 기반 ID → 위경도 복원
    try:
        lat_part, lng_part = id.split("_")
        lat = float(lat_part) / 1_000_000
        lon = float(lng_part) / 1_000_000
    except:
        raise HTTPException(status_code=400, detail="ID 형식 오류")

    # 2) 가장 가까운 station 찾기
    df = df.loc[:, ~df.columns.duplicated()]
    df["distance"] = (df["위도"] - lat)**2 + (df["경도"] - lon)**2
    station = df.loc[df["distance"].idxmin()].to_dict()
    station.pop("distance", None)

    # 3) HTML 생성
    html = terrain_service.generate_interactive_html(lon=lon, lat=lat, radius=500)
    return HTMLResponse(content=html)



@router.get("/{id}", response_model=GasStationResponse)
async def get_station_detail(
    id: str = Path(..., description="좌표 기반 고유 ID (예: 35689819_128445642)"),
    service: GeoService = Depends(get_geo_service),
):
    """
    좌표 기반 고유 ID로 주유소 상세 조회
    """
    try:
        df = service.data.get("gas_station")

        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="주유소 데이터가 비어있습니다.")

        # -------------------------
        # 1) 중복된 위도/경도 컬럼 제거
        # -------------------------
        # station.csv → rename 과정에서 "위도", "경도"가 2개씩 생김 → 이걸 제거해야 distance 계산 가능
        df = df.loc[:, ~df.columns.duplicated()]

        # 필수 컬럼 체크
        if "위도" not in df.columns or "경도" not in df.columns:
            raise HTTPException(status_code=500, detail="위도/경도 컬럼이 누락되었습니다.")

        # -------------------------
        # 2) 좌표 기반 ID 파싱
        # -------------------------
        # 예: "35689819_128445642"
        try:
            lat_part, lng_part = id.split("_")
            lat = float(lat_part) / 1_000_000
            lng = float(lng_part) / 1_000_000
        except Exception:
            raise HTTPException(status_code=400, detail="ID 형식 오류 (예: 35689819_128445642)")

        # -------------------------
        # 3) 가장 가까운 station 찾기
        # -------------------------
        # 거리 계산
        df["distance"] = ((df["위도"] - lat) ** 2 + (df["경도"] - lng) ** 2)

        # 최소 거리 행 선택
        nearest_idx = df["distance"].idxmin()
        station = df.loc[nearest_idx].to_dict()

        # distance 제거
        station.pop("distance", None)

        return JSONResponse(content=station)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상세 조회 오류: {str(e)}")
