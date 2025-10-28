"""
주유소 관련 API 스키마
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import date


class GasStationBase(BaseModel):
    """주유소 기본 스키마"""
    year: Optional[int] = None
    date: Optional[date] = None
    category: Optional[str] = Field(None, description="분류 (주유소 등)")
    status: Optional[str] = Field(None, description="상태 (폐업, 휴업 등)")
    name: Optional[str] = Field(None, description="상호명")
    address: str = Field(..., description="주소")
    exists: Optional[bool] = Field(None, description="잔존여부")
    note: Optional[str] = Field(None, description="비고")
    admin_region: Optional[str] = Field(None, description="행정구역")


class GasStationCreate(GasStationBase):
    """주유소 생성 스키마"""
    pass


class GasStationUpdate(BaseModel):
    """주유소 업데이트 스키마"""
    year: Optional[int] = None
    date: Optional[date] = None
    category: Optional[str] = None
    status: Optional[str] = None
    name: Optional[str] = None
    address: Optional[str] = None
    exists: Optional[bool] = None
    note: Optional[str] = None
    admin_region: Optional[str] = None


class GasStationInDB(GasStationBase):
    """DB에 저장된 주유소 스키마"""
    id: int

    class Config:
        from_attributes = True


class GasStationResponse(GasStationInDB):
    """주유소 응답 스키마"""
    population: Optional[int] = Field(None, description="인구수")
    business_density: Optional[float] = Field(None, description="사업체 밀집도")
    lat: Optional[float] = Field(None, description="위도")
    lng: Optional[float] = Field(None, description="경도")


class GasStationList(BaseModel):
    """주유소 목록 응답 스키마"""
    count: int = Field(..., description="전체 항목 수")
    items: List[GasStationResponse] = Field(..., description="주유소 목록")
