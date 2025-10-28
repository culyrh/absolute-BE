"""
주유소 관련 데이터 모델
"""

from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field


class GasStationBase(BaseModel):
    """주유소 기본 모델"""
    year: Optional[int] = None
    date: Optional[date] = None
    category: Optional[str] = None
    status: Optional[str] = None
    name: Optional[str] = None
    address: str
    exists: Optional[bool] = None
    note: Optional[str] = None
    admin_region: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class GasStation(GasStationBase):
    """주유소 모델 (DB 저장용)"""
    id: int
    population: Optional[int] = None
    business_density: Optional[float] = None

    class Config:
        from_attributes = True


class GasStationCreate(GasStationBase):
    """주유소 생성 모델"""
    pass


class GasStationUpdate(GasStationBase):
    """주유소 업데이트 모델"""
    pass


class GasStationInDB(GasStationBase):
    """데이터베이스에 저장된 주유소 모델"""
    id: int

    class Config:
        from_attributes = True
