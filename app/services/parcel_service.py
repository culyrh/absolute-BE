# app/services/parcel_service.py

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from app.services.geoai_config import GeoAIConfig

_logger = logging.getLogger(__name__)


class ParcelService:
    """지적도(필지) 데이터 조회 유틸리티."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.cfg = GeoAIConfig()
        self.base_dir = base_dir or self.cfg.parcel_base_dir
        self.cache: Dict[str, gpd.GeoDataFrame] = {}
        self._is_loaded = False
        self._last_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and bool(self.cache)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _record_error(self, message: str) -> None:
        self._last_error = message
        _logger.warning(message)

    def load_parcels(self, sidocode: str) -> gpd.GeoDataFrame:
        if sidocode in self.cache:
            return self.cache[sidocode]

        folder = Path(self.base_dir) / sidocode
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"지적도 디렉터리를 찾을 수 없습니다: {folder}")

        shp_files = [f for f in os.listdir(folder) if f.endswith(".shp")]
        if not shp_files:
            raise FileNotFoundError(f"SHP 파일이 존재하지 않습니다: {folder}")

        shp_path = folder / shp_files[0]
        gdf = gpd.read_file(shp_path)
        gdf = gdf.to_crs(epsg=4326)

        self.cache[sidocode] = gdf
        if not gdf.empty:
            self._is_loaded = True
            self._last_error = None

        return gdf

    def _ensure_any_dataset(self) -> None:
        if self.cache:
            return

        if not Path(self.base_dir).exists():
            self._record_error(f"지적도 기본 경로가 존재하지 않습니다: {self.base_dir}")
            return

        for entry in sorted(Path(self.base_dir).iterdir()):
            if entry.is_dir():
                try:
                    self.load_parcels(entry.name)
                    return
                except Exception as exc:
                    self._record_error(f"지적도 로드 실패({entry.name}): {exc}")
                    continue

        if not self.cache:
            self._record_error("사용 가능한 지적도 데이터가 없습니다.")

    def get_nearby_parcels(self, lat: float, lng: float, radius: float = 0.003) -> gpd.GeoDataFrame:
        self._ensure_any_dataset()

        if not self.cache:
            return gpd.GeoDataFrame(columns=["geometry"])

        search_point = Point(lng, lat)
        results = []

        for sidocode, gdf in self.cache.items():
            try:
                nearby = gdf[gdf.geometry.distance(search_point) <= radius]
            except Exception as exc:
                self._record_error(f"필지 거리 계산 실패({sidocode}): {exc}")
                continue

            if not nearby.empty:
                results.append(nearby)

        if not results:
            first_gdf = next(iter(self.cache.values()))
            return gpd.GeoDataFrame(columns=first_gdf.columns)

        concatenated = pd.concat(results, ignore_index=True)
        return gpd.GeoDataFrame(concatenated, geometry="geometry", crs=results[0].crs)


_parcel_service_instance: Optional[ParcelService] = None


def get_parcel_service() -> ParcelService:
    global _parcel_service_instance

    if _parcel_service_instance is None:
        _parcel_service_instance = ParcelService()

    return _parcel_service_instance
