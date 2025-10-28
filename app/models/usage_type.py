"""
용도 유형 관련 데이터 모델
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class UsageTypeBase(BaseModel):
    """용도 유형 기본 모델"""
    name: str
    description: Optional[str] = None
    examples: Optional[List[str]] = None


class UsageType(UsageTypeBase):
    """용도 유형 모델 (DB 저장용)"""
    id: int
    population_norm: Optional[float] = None
    traffic_norm: Optional[float] = None
    tourism_norm: Optional[float] = None
    business_density_norm: Optional[float] = None
    land_price_norm: Optional[float] = None
    
    class Config:
        from_attributes = True


class UsageTypeCreate(UsageTypeBase):
    """용도 유형 생성 모델"""
    pass


class UsageTypeUpdate(UsageTypeBase):
    """용도 유형 업데이트 모델"""
    pass


class UsageTypeInDB(UsageTypeBase):
    """데이터베이스에 저장된 용도 유형 모델"""
    id: int

    class Config:
        from_attributes = True


class UsageTypeCentroid(BaseModel):
    """용도 유형 센트로이드 모델 (대분류 센터로이드)"""
    usage_type: str
    population_norm: Optional[float] = None
    traffic_norm: Optional[float] = None
    tourism_norm: Optional[float] = None
    business_density_norm: Optional[float] = None
    land_price_norm: Optional[float] = None
    region: Optional[str] = None
