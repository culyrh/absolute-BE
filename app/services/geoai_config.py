# app/services/geoai_config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# 프로젝트 루트 기준 (absolute-be/)
BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass
class GeoAIConfig:
    """
    GeoAI 공통 설정
    """

    # 기본 디렉토리들
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"

    # 지적도 기본 디렉토리 (시도 코드별 하위 폴더)
    # 예: data/parcels/45/*.shp (전북)
    parcel_base_dir: Path = data_dir / "parcels"

    # CSV 경로들
    station_csv: Path = data_dir / "station.csv"
    train_csv: Path = data_dir / "train.csv"
    test_csv: Path = data_dir / "test_data.csv"

    # 좌표계
    wgs84_crs: str = "EPSG:4326"   # 위/경도
    metric_crs: str = "EPSG:5186"  # 거리/면적 계산용 (한국 TM 계열)

    # 버퍼 반경 (미터)
    buffer_small_m: float = 300.0
    buffer_large_m: float = 500.0
