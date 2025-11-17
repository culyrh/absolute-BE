# app/services/geoai_config.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from app.config import BASE_DIR, DATA_DIR


@dataclass
class GeoAIConfig:
    """
    GeoAI 공통 설정
    """

    # 기본 디렉토리
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR

    # 지적도 기본 디렉토리
    parcel_base_dir: Path = data_dir / "parcels"

    # CSV 파일 경로 (절대경로)
    station_csv: Path = data_dir / "station.csv"
    train_csv: Path = data_dir / "train.csv"
    test_csv: Path = data_dir / "test_data.csv"

    # 좌표계
    wgs84_crs: str = "EPSG:4326"
    metric_crs: str = "EPSG:5186"

    # 버퍼 반경
    buffer_small_m: float = 300.0
    buffer_large_m: float = 500.0
