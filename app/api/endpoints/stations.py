"""
주유소 정보 관련 API 엔드포인트
"""

from collections import Counter
from html import escape
from typing import Optional, List, Dict, Any

import traceback
import pandas as pd
import folium
import math
from fastapi import APIRouter, Depends, Query, HTTPException, Path
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from shapely.geometry import Point

from app.api.dependencies import get_geo_service, get_report_service
from app.schemas.gas_station import GasStationList, GasStationResponse
from app.services.geo_service import GeoService
from app.services.ml_location_recommender import MLLocationRecommender
from app.services.parcel_service import get_parcel_service
from app.services.recommend_service import RecommendationService, get_recommendation_service
from app.services.report_service import LLMReportService


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


def _extract_ml_recommendations(station: Dict[str, Any]) -> List[Dict[str, Any]]:
    """station 데이터에 포함된 recommend1~3 컬럼을 표준 추천 포맷으로 변환한다."""

    recommendations: List[Dict[str, Any]] = []
    for rank in range(1, 4):
        value = station.get(f"recommend{rank}")
        if value is None:
            continue

        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue

        recommendations.append({
            "type": text,
            "rank": rank,
            "source": "ml_recommend",
            "description": f"ML 기반 추천 순위 {rank}위",
        })

    return recommendations


_ml_recommender: Optional[MLLocationRecommender] = None


def _get_ml_recommender() -> Optional[MLLocationRecommender]:
    global _ml_recommender

    if _ml_recommender is not None:
        return _ml_recommender

    try:
        instance = MLLocationRecommender()
        instance.train()
        _ml_recommender = instance
        return _ml_recommender
    except Exception as exc:
        print(f"MLLocationRecommender 초기화 실패: {exc}")
        return None


def _live_ml_recommendations(station: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    """실시간 ML 추천(top-N)을 호출해 표준 포맷으로 반환한다."""

    recommender = _get_ml_recommender()
    if recommender is None:
        return []

    keyword = (
        station.get("상호")
        or station.get("상호명")
        or station.get("업체명")
        or station.get("주소")
        or station.get("지번주소")
    )
    if not keyword:
        return []

    try:
        result = recommender.recommend_for_station(str(keyword), top_n=top_n)
    except Exception as exc:
        print(f"실시간 ML 추천 실패: {exc}")
        return []

    items = result.get("results") or []
    formatted: List[Dict[str, Any]] = []
    for item in items:
        category = item.get("category")
        if not category:
            continue
        formatted.append(
            {
                "type": category,
                "rank": item.get("rank"),
                "probability": item.get("probability"),
                "source": "ml_recommend",
                "description": f"ML 기반 추천 순위 {item.get('rank')}위",
            }
        )

    return formatted


USAGE_EXAMPLES: Dict[str, List[str]] = {
    "가설건축": ["모듈러 임시 판매존", "이벤트·전시 팝업", "가변형 임대 공간"],
    "공동주택": ["도심형 소형 주택", "코리빙 레지던스", "청년 주거 특화"],
    "공장": ["경량 조립·패키징", "스마트 마이크로 팩토리", "지역 특화 생산 거점"],
    "근린생활시설": ["카페·베이커리", "드라이브 스루 매장", "키즈·펫 프렌들리 커뮤니티"],
    "기타": ["지역 커뮤니티 허브", "생활 편의 복합 공간", "공공·민간 협력 거점"],
    "숙박시설": ["스마트 체류형 숙소", "마이크로 호텔", "관광·MICE 연계 숙박"],
    "업무시설": ["스타트업 스튜디오", "라이트 오피스·회의실", "공공·민간 합동 거점"],
    "자동차관련시설": ["EV 급속·완속 복합 충전소", "프리미엄 세차·디테일링", "모빌리티 공유 거점"],
    "판매시설": ["편의형 슈퍼마켓", "지역 특화 리테일", "팝업 스토어 존"],
}


def _usage_examples(usage_type: str) -> List[str]:
    for keyword, examples in USAGE_EXAMPLES.items():
        if keyword in usage_type:
            return examples
    return ["복합 커뮤니티 라운지", "지역 맞춤형 서비스 존", "공공·민간 협력형 파일럿"]


def _merge_recommendations(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """추천 항목을 용도명 기준으로 병합한다."""

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for source_list in (primary, secondary):
        for item in source_list:
            usage = item.get("type") or item.get("usage_type") or item.get("category")
            if not usage:
                continue

            usage_key = str(usage).strip()
            if not usage_key or usage_key.lower() == "nan" or usage_key in seen:
                continue

            merged.append(item)
            seen.add(usage_key)

            if len(merged) >= limit:
                return merged

    return merged


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


@router.get("/{id}/statics")
async def get_station_statics(
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

        if adm_raw is None:
            raise HTTPException(status_code=500, detail="station adm_cd2 없음")

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
        }

        # -------------------------------------------
        # 11) 백분위 계산
        # -------------------------------------------
        def percentile(series, value):
            if value is None:
                return None
            arr = series.dropna().values
            if len(arr) == 0:
                return None
            return float((arr < value).mean() * 100)

        percentiles = {
            name: percentile(region_train[tr_col], metrics[name])
            for name, (st_col, tr_col) in FEATURE_COLS.items()
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
    recommend_service: RecommendationService = Depends(get_recommendation_service),
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

        # 2. 추천 결과 (ML recommend1~3 + 서비스 추천 병합)
        try:
            recommendations = recommend_service.recommend_by_query(address, top_k=5)
            rec_items = recommendations.get('items', [])
        except Exception as rec_error:
            print(f"추천 서비스 오류: {rec_error}")
            rec_items = []

        live_ml_rec_items = _live_ml_recommendations(station, top_n=3)
        static_ml_rec_items = _extract_ml_recommendations(station)
        primary_recs = live_ml_rec_items or static_ml_rec_items
        combined_recommendations = _merge_recommendations(primary_recs, rec_items, limit=5)

        parcel_summary = None

        # 3. 지도 생성
        m = folium.Map(location=[lat, lng], zoom_start=17, tiles='OpenStreetMap')

        try:
            parcel_service = get_parcel_service()
            nearby_parcels = parcel_service.get_nearby_parcels(lat, lng, radius=0.003)
            parcel_summary = _summarise_nearby_parcels(nearby_parcels, lat, lng)
        except Exception as parcel_error:
            print(f"지적도 서비스 오류: {parcel_error}")
            nearby_parcels = None

        llm_report = await report_service.generate_report(
            station,
            combined_recommendations,
            parcel_summary=parcel_summary,
            station_id=station_id
        )

        if nearby_parcels is not None and not nearby_parcels.empty:
            # 필지별로 그리기 (최대 200개)
            for idx, row in nearby_parcels.head(200).iterrows():
                # 면적 계산
                area = row.geometry.area * (111000 ** 2)

                # 크기별 색상
                if area < 300:
                    color = '#3498db'  # 파랑
                    label = '소형'
                elif area < 1000:
                    color = '#2ecc71'  # 초록
                    label = '중형'
                elif area < 3000:
                    color = '#f39c12'  # 주황
                    label = '대형'
                else:
                    color = '#e74c3c'  # 빨강
                    label = '초대형'

                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, c=color: {
                        'fillColor': c,
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.4
                    },
                    tooltip=f"{label} - {row.get('JIBUN', 'N/A')} - {area:.0f}㎡"
                ).add_to(m)
        
        # 3-2. 주유소 마커
        popup_html = f"""
        <div style='white-space: normal; width: 260px; line-height: 1.4;'>
            <div style='font-weight: 600; margin-bottom: 4px;'>{escape(str(name))}</div>
            <div>{escape(str(address))}</div>
        </div>
        """
        folium.Marker(
            [lat, lng],
            popup=folium.Popup(popup_html, max_width=320, min_width=220),
            tooltip=name,
            icon=folium.Icon(color='red', icon='gas-pump', prefix='fa')
        ).add_to(m)
        
        # 3-3. 반경 표시
        folium.Circle(
            [lat, lng],
            radius=300,
            color='red',
            fill=True,
            fillOpacity=0.1,
            popup='반경 300m'
        ).add_to(m)
        
        # 범례 추가
        legend_html = '''
        <div style="position: absolute; bottom: 20px; left: 20px;
                    background: rgba(255, 255, 255, 0.95); padding: 12px 16px; border: 1px solid #ccc;
                    border-radius: 5px; z-index: 500; font-size: 13px; line-height: 1.4;">
            <p style="margin: 0 0 10px 0; font-weight: bold;">필지 크기</p>
            <p style="margin: 5px 0;">
                <span style="background: #3498db; padding: 3px 10px;">　</span> 소형 (&lt;300㎡)
            </p>
            <p style="margin: 5px 0;">
                <span style="background: #2ecc71; padding: 3px 10px;">　</span> 중형 (300-1000㎡)
            </p>
            <p style="margin: 5px 0;">
                <span style="background: #f39c12; padding: 3px 10px;">　</span> 대형 (1000-3000㎡)
            </p>
            <p style="margin: 5px 0;">
                <span style="background: #e74c3c; padding: 3px 10px;">　</span> 초대형 (&gt;3000㎡)
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        map_html = m._repr_html_()
        
        # 4. LLM 분석 결과 HTML
        analysis_sections = []
        summary_text = llm_report.get('summary') if isinstance(llm_report, dict) else None
        insights_list = llm_report.get('insights', []) if isinstance(llm_report, dict) else []
        actions_list = llm_report.get('actions', []) if isinstance(llm_report, dict) else []

        if summary_text:
            analysis_sections.append(f"<p style=\"line-height: 1.6;\">{summary_text}</p>")

        if insights_list:
            insights_items = ''.join(
                f"<li style=\"margin-bottom: 6px;\">{insight}</li>" for insight in insights_list
            )
            analysis_sections.append(
                "<div><h3 style=\"margin-bottom: 8px; color: #2c3e50;\">핵심 인사이트</h3>"
                f"<ul style=\"padding-left: 20px; margin-top: 0;\">{insights_items}</ul></div>"
            )

        if actions_list:
            actions_items = ''.join(
                f"<li style=\"margin-bottom: 6px;\">{action}</li>" for action in actions_list
            )
            analysis_sections.append(
                "<div><h3 style=\"margin-bottom: 8px; color: #2c3e50;\">권장 실행 항목</h3>"
                f"<ol style=\"padding-left: 20px; margin-top: 0;\">{actions_items}</ol></div>"
            )

        if not analysis_sections:
            analysis_sections.append(
                "<p style=\"color: #7f8c8d;\">LLM 분석 결과를 가져오지 못했습니다. 기본 정보를 참고하세요.</p>"
            )

        llm_analysis_html = "".join(analysis_sections)

        # 5. 추천 결과 HTML
        recommendations_html = ""
        highlight_cards = ""

        for i, item in enumerate(combined_recommendations[:5], 1):
            score = item.get('score') or item.get('probability') or item.get('similarity')
            try:
                score_display = f"{float(score):.3f}" if score is not None else "-"
            except (TypeError, ValueError):
                score_display = str(score)

            description = item.get('description', '')
            item_type = item.get('type', item.get('usage_type', item.get('category', '제안 항목')))
            source = item.get('source', '추천')
            recommendations_html += f"""
            <div class=\"rec-card\">
                <div class=\"rec-rank\">{i}</div>
                <div class=\"rec-body\">
                    <div class=\"rec-title\">{item_type}</div>
                    <div class=\"rec-meta\">사용한 알고리즘: {source} · 점수/확률: {score_display}</div>
                    <div class=\"rec-desc\">{description or '요약 정보가 추가될 예정입니다.'}</div>
                    <div class=\"rec-chips\">
                        {''.join(f'<span class="chip">{ex}</span>' for ex in _usage_examples(str(item_type)))}
                    </div>
                </div>
            </div>
            """

            if i <= 3:
                highlight_cards += f"""
                <div class=\"highlight-card\">
                    <div class=\"highlight-rank\">Top {i}</div>
                    <div class=\"highlight-title\">{item_type}</div>
                    <p class=\"highlight-desc\">{description or '상위 추천 활용 방안을 우선 검토하세요.'}</p>
                    <div class=\"rec-chips\">
                        {''.join(f'<span class="chip ghost">{ex}</span>' for ex in _usage_examples(str(item_type)))}
                    </div>
                </div>
                """

        if not recommendations_html:
            recommendations_html = "<p class=\"muted\">추천 데이터를 찾을 수 없습니다.</p>"

        # 6. HTML 조합
        html = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="utf-8">
            <title>{name} 입지 분석 보고서</title>
            <style>
                :root {{
                    --bg: #ecf4ee;
                    --card: #ffffff;
                    --accent: #2fb36f;
                    --accent-2: #1f9255;
                    --text: #10291a;
                    --muted: #5f7263;
                    --border: #d9e7dc;
                }}
                * {{ box-sizing: border-box; }}
                body {{
                    font-family: 'Noto Sans KR', 'Pretendard', Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: radial-gradient(circle at 20% 20%, rgba(47,179,111,0.10), transparent 32%),
                                radial-gradient(circle at 80% 0%, rgba(31,146,85,0.08), transparent 32%),
                                var(--bg);
                    color: var(--text);
                    line-height: 1.6;
                }}
                .page {{
                    max-width: 1180px;
                    margin: 32px auto;
                    padding: 8px 18px 42px;
                }}
                .hero {{
                    background: linear-gradient(135deg, #2fb36f, #1f9255);
                    color: white;
                    border-radius: 18px;
                    padding: 28px 32px;
                    box-shadow: 0 18px 40px rgba(31, 146, 85, 0.30);
                }}
                .hero h1 {{ margin: 0 0 6px 0; font-size: 30px; }}
                .hero p {{ margin: 0; color: rgba(255,255,255,0.9); }}
                .section {{ margin-top: 20px; }}
                .section-title {{
                    font-size: 19px;
                    margin-bottom: 12px;
                    color: #0f172a;
                    letter-spacing: -0.02em;
                }}
                .card {{
                    background: var(--card);
                    border-radius: 14px;
                    padding: 18px 20px;
                    box-shadow: 0 12px 30px rgba(17, 24, 39, 0.06);
                    border: 1px solid var(--border);
                }}
                .glass {{
                    background: linear-gradient(135deg, rgba(47,179,111,0.08), rgba(31,146,85,0.06));
                    border: 1px solid rgba(255,255,255,0.35);
                }}
                .map-container {{
                    height: 520px;
                    border-radius: 14px;
                    overflow: hidden;
                    border: 1px solid var(--border);
                }}
                .map-note {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
                .muted {{ color: var(--muted); }}
                .grid {{ display: grid; gap: 14px; }}
                .grid.two {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
                .rec-card {{
                    display: grid;
                    grid-template-columns: 62px 1fr;
                    gap: 14px;
                    padding: 14px;
                    border-radius: 12px;
                    background: #f4fbf6;
                    border: 1px solid var(--border);
                }}
                .rec-rank {{
                    width: 48px; height: 48px;
                    border-radius: 12px;
                    background: linear-gradient(135deg, #2fb36f, #1f9255);
                    color: white;
                    display: grid;
                    place-items: center;
                    font-weight: 700;
                    font-size: 18px;
                }}
                .rec-title {{ font-weight: 700; font-size: 17px; color: #0f172a; margin-bottom: 2px; }}
                .rec-meta {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
                .rec-desc {{ color: #27303f; font-size: 14px; margin-bottom: 10px; }}
                .rec-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
                .chip {{
                    background: #e6f7ed;
                    color: #1f7a4c;
                    padding: 6px 10px;
                    border-radius: 999px;
                    font-size: 12px;
                    border: 1px solid #b2e3c6;
                }}
                .chip.ghost {{
                    background: rgba(47,179,111,0.10);
                    color: #1f7a4c;
                    border-color: rgba(47,179,111,0.25);
                }}
                .highlight-wrap {{
                    background: linear-gradient(135deg, rgba(47,179,111,0.10), rgba(31,146,85,0.10));
                    border: 1px solid rgba(47,179,111,0.20);
                    border-radius: 14px;
                    padding: 16px;
                }}
                .highlight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
                .highlight-card {{
                    background: white;
                    border-radius: 12px;
                    padding: 14px;
                    border: 1px solid var(--border);
                    box-shadow: 0 12px 30px rgba(17, 24, 39, 0.04);
                }}
                .highlight-rank {{ font-size: 12px; font-weight: 700; color: #1f7a4c; letter-spacing: 0.02em; }}
                .highlight-title {{ font-weight: 700; font-size: 16px; margin: 4px 0; color: #0f2918; }}
                .highlight-desc {{ margin: 0; color: #274231; font-size: 14px; }}
                .analysis-block p {{ margin: 0 0 10px 0; }}
                .analysis-block h3 {{ margin: 14px 0 6px; }}
            </style>
        </head>
        <body>
            <div class="page">
                <div class="hero">
                    <h1>📍 {name}</h1>
                    <p>{address}</p>
                </div>

                <div class="section grid two">
                    <div class="card">
                        <div class="section-title">🗺️ 위치 및 필지 지도</div>
                        <div class="map-container">{map_html}</div>
                        <p class="map-note">
                            색상은 필지 크기를 나타내며, 붉은 원은 반경 300m 범위를 의미합니다.
                        </p>
                    </div>

                    <div class="card glass analysis-block">
                        <div class="section-title">🤖 LLM 기반 분석 요약</div>
                        {llm_analysis_html}
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">✨ 상위 3개 활용 방안 브리핑</div>
                    <div class="highlight-wrap">
                        <div class="highlight-grid">
                            {highlight_cards or '<p class="muted">추천 데이터를 찾을 수 없습니다.</p>'}
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">💡 추천 활용방안 상세</div>
                    <div class="grid">
                        {recommendations_html}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"보고서 생성 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cases", response_model=Dict[str, Any])
async def get_station_cases():
    """
    활용 사례 카드 API
    
    폐주유소의 다양한 활용 사례 정보를 카드 형태로 제공합니다.
    """
    try:
        # 대분류 정보 활용한 활용 사례 카드
        cases = [
            {
                "id": 1,
                "title": "근린생활시설",
                "description": "일상생활에 필요한 서비스를 제공하는 시설로 활용",
                "image_url": "/assets/cases/convenience.jpg"
            },
            {
                "id": 2,
                "title": "공동주택",
                "description": "주거 공간으로 재활용하여 주택 공급에 기여",
                "image_url": "/assets/cases/housing.jpg"
            },
            {
                "id": 3,
                "title": "자동차관련시설",
                "description": "전기차 충전소나 정비소로 전환하여 활용",
                "image_url": "/assets/cases/automotive.jpg"
            },
            {
                "id": 4,
                "title": "판매시설",
                "description": "소매점이나 마켓으로 활용하여 지역 상권 활성화",
                "image_url": "/assets/cases/retail.jpg"
            },
            {
                "id": 5,
                "title": "업무시설",
                "description": "코워킹 스페이스나 사무실로 활용",
                "image_url": "/assets/cases/office.jpg"
            }
        ]
        
        # 캐싱 헤더 설정 (1일)
        headers = {"Cache-Control": "public, max-age=86400"}
        
        return JSONResponse(
            content={"count": len(cases), "items": cases},
            headers=headers
        )
    except Exception as e:
        print(f"활용 사례 카드 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"활용 사례 카드 조회 중 오류가 발생했습니다: {str(e)}")


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

