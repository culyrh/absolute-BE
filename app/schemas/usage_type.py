"""
용도 유형 관련 API 스키마
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class UsageTypeBase(BaseModel):
    """용도 유형 기본 스키마"""
    name: str = Field(..., description="용도 유형명 (대분류)")
    description: Optional[str] = Field(None, description="설명")
    examples: Optional[List[str]] = Field(None, description="예시")


class UsageTypeCreate(UsageTypeBase):
    """용도 유형 생성 스키마"""
    pass


class UsageTypeUpdate(BaseModel):
    """용도 유형 업데이트 스키마"""
    name: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[str]] = None


class UsageTypeInDB(UsageTypeBase):
    """DB에 저장된 용도 유형 스키마"""
    id: int

    class Config:
        from_attributes = True


class UsageTypeVectorBase(BaseModel):
    """용도 유형 벡터 기본 스키마"""
    population_norm: Optional[float] = Field(None, description="인구수 정규화 값")
    traffic_norm: Optional[float] = Field(None, description="교통량 정규화 값")
    tourism_norm: Optional[float] = Field(None, description="숙박업소(관광지수) 정규화 값")
    business_density_norm: Optional[float] = Field(None, description="상권밀집도 정규화 값")
    land_price_norm: Optional[float] = Field(None, description="공시지가 정규화 값")


class UsageTypeResponse(UsageTypeInDB, UsageTypeVectorBase):
    """용도 유형 응답 스키마"""
    pass


class UsageTypeList(BaseModel):
    """용도 유형 목록 응답 스키마"""
    count: int = Field(..., description="전체 항목 수")
    items: List[UsageTypeResponse] = Field(..., description="용도 유형 목록")


class UsageTypeCentroid(BaseModel):
    """용도 유형 센트로이드 스키마"""
    usage_type: str = Field(..., description="용도 유형명 (대분류)")
    region: str = Field(..., description="권역")
    population_norm: Optional[float] = Field(None, description="인구수 정규화 값")
    traffic_norm: Optional[float] = Field(None, description="교통량 정규화 값")
    tourism_norm: Optional[float] = Field(None, description="숙박업소(관광지수) 정규화 값")
    business_density_norm: Optional[float] = Field(None, description="상권밀집도 정규화 값")
    land_price_norm: Optional[float] = Field(None, description="공시지가 정규화 값")


class UsageTypeCentroidList(BaseModel):
    """용도 유형 센트로이드 목록 응답 스키마"""
    count: int = Field(..., description="전체 항목 수")
    items: List[UsageTypeCentroid] = Field(..., description="용도 유형 센트로이드 목록")
