"""
데이터 전처리 유틸리티
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from app.utils.data_loader import find_column_by_keyword


def extract_admin_region(address: str) -> Optional[str]:
    """주소에서 행정구역(시/군/구) 추출"""
    if not isinstance(address, str):
        return None
    
    matches = re.findall(r"[가-힣]+시|[가-힣]+군|[가-힣]+구", address)
    if matches:
        return matches[0]
    return None


def extract_province(address: str) -> Optional[str]:
    """주소에서 도/광역시 추출"""
    if not isinstance(address, str):
        return None
    
    # 특별시, 광역시, 특별자치시, 도, 특별자치도 패턴 
    patterns = [
        r"서울특별시|서울시",
        r"부산광역시|부산시", 
        r"대구광역시|대구시", 
        r"인천광역시|인천시",
        r"광주광역시|광주시", 
        r"대전광역시|대전시", 
        r"울산광역시|울산시",
        r"세종특별자치시|세종시",
        r"경기도", 
        r"강원특별자치도|강원도",
        r"충청북도|충북",
        r"충청남도|충남",
        r"전라북도|전북",
        r"전라남도|전남",
        r"경상북도|경북",
        r"경상남도|경남",
        r"제주특별자치도|제주도"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, address)
        if matches:
            # 정규화된 이름으로 반환
            if "서울" in matches[0]:
                return "서울특별시"
            elif "부산" in matches[0]:
                return "부산광역시"
            elif "대구" in matches[0]:
                return "대구광역시"
            elif "인천" in matches[0]:
                return "인천광역시"
            elif "광주" in matches[0]:
                return "광주광역시"
            elif "대전" in matches[0]:
                return "대전광역시"
            elif "울산" in matches[0]:
                return "울산광역시"
            elif "세종" in matches[0]:
                return "세종특별자치시"
            elif "경기" in matches[0]:
                return "경기도"
            elif "강원" in matches[0]:
                return "강원특별자치도"
            elif "충북" in matches[0] or "충청북" in matches[0]:
                return "충청북도"
            elif "충남" in matches[0] or "충청남" in matches[0]:
                return "충청남도"
            elif "전북" in matches[0] or "전라북" in matches[0]:
                return "전라북도"
            elif "전남" in matches[0] or "전라남" in matches[0]:
                return "전라남도"
            elif "경북" in matches[0] or "경상북" in matches[0]:
                return "경상북도"
            elif "경남" in matches[0] or "경상남" in matches[0]:
                return "경상남도"
            elif "제주" in matches[0]:
                return "제주특별자치도"
    
    return None


def normalize_region(region: str) -> str:
    """권역명 정규화"""
    if not isinstance(region, str):
        return "기타"
    
    # 세종특별자치시는 충청남도로 포함
    if "세종" in region:
        return "충청남도"
    
    # 도/광역시 정규화
    if "서울" in region:
        return "서울특별시"
    elif "부산" in region:
        return "부산광역시"
    elif "대구" in region:
        return "대구광역시"
    elif "인천" in region:
        return "인천광역시"
    elif "광주" in region:
        return "광주광역시"
    elif "대전" in region:
        return "대전광역시"
    elif "울산" in region:
        return "울산광역시"
    elif "경기" in region:
        return "경기도"
    elif "강원" in region:
        return "강원특별자치도"
    elif "충북" in region or "충청북" in region:
        return "충청북도"
    elif "충남" in region or "충청남" in region:
        return "충청남도"
    elif "전북" in region or "전라북" in region:
        return "전라북도"
    elif "전남" in region or "전라남" in region:
        return "전라남도"
    elif "경북" in region or "경상북" in region:
        return "경상북도"
    elif "경남" in region or "경상남" in region:
        return "경상남도"
    elif "제주" in region:
        return "제주특별자치도"
    
    return region


def preprocess_gas_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """주유소 데이터 전처리"""
    # 데이터 복사
    processed_df = df.copy()
    
    # 주소 컬럼 탐색
    address_col = find_column_by_keyword(processed_df, ["주소", "소재지", "지번", "address"])
    if not address_col:
        raise ValueError("주소 컬럼을 찾을 수 없습니다.")
    
    # 행정구역 추출
    processed_df["행정구역"] = processed_df[address_col].apply(extract_admin_region)
    
    # 권역(도/광역시) 추출
    processed_df["권역"] = processed_df[address_col].apply(extract_province)
    
    # 결측치 처리
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 컬럼명 통일
    if address_col != "주소":
        processed_df.rename(columns={address_col: "주소"}, inplace=True)
    
    return processed_df


def merge_with_stats(gas_df: pd.DataFrame, pop_df: pd.DataFrame, biz_df: pd.DataFrame) -> pd.DataFrame:
    """주유소 데이터에 인구수와 사업체 데이터 병합"""
    # 인구수 컬럼 찾기
    pop_col = find_column_by_keyword(pop_df, ["인구"])
    if not pop_col:
        raise ValueError("인구수 컬럼을 찾을 수 없습니다.")
    
    # 사업체 컬럼 찾기
    biz_col = find_column_by_keyword(biz_df, ["사업체", "천명"])
    if not biz_col:
        raise ValueError("사업체수 컬럼을 찾을 수 없습니다.")
    
    # 병합할 수 있는 키 확인
    if "행정구역" not in gas_df.columns:
        raise ValueError("주유소 데이터에 행정구역 컬럼이 없습니다.")
    
    # 데이터 병합
    merged_df = gas_df.copy()
    
    # 인구수 데이터 병합
    if "행정구역" in pop_df.columns:
        merged_df = merged_df.merge(
            pop_df[["행정구역", pop_col]], 
            on="행정구역", 
            how="left"
        )
    
    # 사업체 데이터 병합
    if "행정구역" in biz_df.columns:
        merged_df = merged_df.merge(
            biz_df[["행정구역", biz_col]], 
            on="행정구역", 
            how="left"
        )
    
    # 컬럼명 통일
    merged_df.rename(
        columns={
            pop_col: "인구[명]",
            biz_col: "인구천명당사업체수",
        },
        inplace=True,
    )
    
    # 결측치 처리
    merged_df.fillna({
        "인구[명]": merged_df["인구[명]"].mean(),
        "인구천명당사업체수": merged_df["인구천명당사업체수"].mean()
    }, inplace=True)
    
    return merged_df


def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """특징 컬럼 정규화"""
    normalized_df = df.copy()
    
    for col in feature_cols:
        if col in df.columns:
            # 결측치 처리
            normalized_df[col].fillna(normalized_df[col].mean(), inplace=True)
            
            # 표준화 (Z-score)
            mean = normalized_df[col].mean()
            std = normalized_df[col].std()
            
            # 0으로 나누기 방지
            if std == 0:
                std = 1e-9
                
            normalized_df[f"{col}_norm"] = (normalized_df[col] - mean) / std
    
    return normalized_df


def categorize_by_usage_type_and_region(df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """용도 유형과 권역별로 데이터 분류"""
    if "대분류" not in df.columns or "권역" not in df.columns:
        raise ValueError("데이터에 '대분류' 또는 '권역' 컬럼이 없습니다.")
    
    result = {}
    
    # 모든 대분류 값 추출
    usage_types = df["대분류"].dropna().unique()
    
    for usage_type in usage_types:
        result[usage_type] = {}
        
        # 해당 대분류의 데이터
        usage_df = df[df["대분류"] == usage_type]
        
        # 모든 권역 값 추출
        regions = usage_df["권역"].dropna().unique()
        
        for region in regions:
            # 해당 대분류와 권역의 데이터
            region_df = usage_df[usage_df["권역"] == region]
            
            if not region_df.empty:
                result[usage_type][region] = region_df
    
    return result


def calculate_centroids(grouped_data: Dict[str, Dict[str, pd.DataFrame]], 
                       feature_cols: List[str],
                       method: str = "mean") -> pd.DataFrame:
    """용도 유형과 권역별 중심점 계산"""
    centroids = []
    
    for usage_type, regions in grouped_data.items():
        for region, df in regions.items():
            centroid = {"usage_type": usage_type, "region": region}
            
            for col in feature_cols:
                if col in df.columns:
                    # 평균 또는 중위값 계산
                    if method == "mean":
                        value = df[col].mean()
                    else:  # median
                        value = df[col].median()
                    
                    centroid[col] = value
            
            centroids.append(centroid)
    
    return pd.DataFrame(centroids)
