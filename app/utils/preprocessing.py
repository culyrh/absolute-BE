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
            elif "강원특별자치도" in matches[0]:
                return "강원특별자치도"
            elif "강원" in matches[0]:
                return "강원특별자치도"
            elif "제주특별자치도" in matches[0]:
                return "제주특별자치도"
            elif "제주" in matches[0]:
                return "제주특별자치도"
            return matches[0]
    
    # 약어로 된 경우
    short_patterns = {
        r"서울": "서울특별시",
        r"부산": "부산광역시",
        r"대구": "대구광역시",
        r"인천": "인천광역시",
        r"광주": "광주광역시",
        r"대전": "대전광역시",
        r"울산": "울산광역시",
        r"세종": "세종특별자치시",
        r"경기": "경기도",
        r"강원": "강원특별자치도",
        r"충북": "충청북도",
        r"충남": "충청남도",
        r"전북": "전라북도",
        r"전남": "전라남도",
        r"경북": "경상북도",
        r"경남": "경상남도",
        r"제주": "제주특별자치도"
    }
    
    for pattern, full_name in short_patterns.items():
        if re.search(f"^{pattern}", address):
            return full_name
    
    return None


def normalize_region(region: str) -> str:
    """지역명 정규화"""
    if not isinstance(region, str):
        return ""
    
    # 특별시, 광역시 정규화
    region_map = {
        "서울": "서울특별시",
        "서울시": "서울특별시",
        "부산": "부산광역시",
        "부산시": "부산광역시",
        "대구": "대구광역시",
        "대구시": "대구광역시",
        "인천": "인천광역시",
        "인천시": "인천광역시",
        "광주": "광주광역시",
        "광주시": "광주광역시",
        "대전": "대전광역시",
        "대전시": "대전광역시",
        "울산": "울산광역시",
        "울산시": "울산광역시",
        "세종": "세종특별자치시",
        "세종시": "세종특별자치시",
        "경기": "경기도",
        "강원": "강원특별자치도",
        "강원도": "강원특별자치도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전라북도",
        "전북특별자치도": "전라북도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주특별자치도",
        "제주도": "제주특별자치도"
    }
    
    for key, value in region_map.items():
        if region == key:
            return value
    
    return region


def preprocess_gas_station_data(df: pd.DataFrame) -> pd.DataFrame:
    """주유소 데이터 전처리"""
    # 컬럼명 찾기
    address_col = find_column_by_keyword(df, ["주소", "소재지"])
    if not address_col:
        raise ValueError("주소 컬럼을 찾을 수 없습니다.")
    
    name_col = find_column_by_keyword(df, ["상호", "명칭", "field5"])  # field5에 상호명 있음
    if not name_col:
        name_col = "상호"  # 기본값 설정
    
    # 데이터 복사 (원본 보존)
    processed_df = df.copy()
    
    # 행정구역 - data_loader에서 만든 행정구역이 있으면 덮어쓰지 않음
    if "행정구역" not in processed_df.columns:
        processed_df["행정구역"] = processed_df["주소"].apply(extract_admin_region)
    
    # 권역 - data_loader에서 만든 권역이 있으면 덮어쓰지 않음
    if "권역" not in processed_df.columns:
        processed_df["권역"] = processed_df["주소"].apply(extract_province)

    # 위도/경도 float 변환 (거리계산 위해 필수)
    for col in ["위도", "경도"]:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")
            
    # 결측치 처리
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 컬럼명 통일
    if address_col != "주소":
        processed_df.rename(columns={address_col: "주소"}, inplace=True)
    
    return processed_df


def merge_with_stats(gas_df: pd.DataFrame, pop_df: pd.DataFrame, biz_df: pd.DataFrame) -> pd.DataFrame:
    """주유소 데이터에 인구수와 사업체 데이터 병합"""
    # 데이터프레임 복사
    merged_df = gas_df.copy()
    
    # 인구수 컬럼 찾기 (여러 패턴 시도)
    pop_col = find_column_by_keyword(pop_df, ["인구", "총인구", "field1"])
    if not pop_col:
        # 인구수 컬럼을 찾을 수 없는 경우 기본값 추가
        print("⚠️ 인구수 컬럼을 찾을 수 없어 기본값을 사용합니다.")
        merged_df["인구[명]"] = 10000  # 기본값 설정
    else:
        # 병합할 수 있는 키 확인
        if "행정구역" not in gas_df.columns:
            print("⚠️ 주유소 데이터에 행정구역 컬럼이 없어 인구 데이터를 병합할 수 없습니다.")
            merged_df["인구[명]"] = 10000  # 기본값 설정
        # 인구수 데이터 병합
        elif "행정구역" in pop_df.columns:
            merged_df = merged_df.merge(
                pop_df[["행정구역", pop_col]], 
                on="행정구역", 
                how="left"
            )
            # 컬럼명 통일
            merged_df.rename(columns={pop_col: "인구[명]"}, inplace=True)
    
    # 사업체 컬럼 찾기
    biz_col = find_column_by_keyword(biz_df, ["사업체", "천명", "field1"])
    if not biz_col:
        # 사업체수 컬럼을 찾을 수 없는 경우 기본값 추가
        print("⚠️ 사업체수 컬럼을 찾을 수 없어 기본값을 사용합니다.")
        merged_df["인구천명당사업체수"] = 80  # 기본값 설정
    else:
        # 병합할 수 있는 키 확인
        if "행정구역" not in gas_df.columns:
            print("⚠️ 주유소 데이터에 행정구역 컬럼이 없어 사업체 데이터를 병합할 수 없습니다.")
            merged_df["인구천명당사업체수"] = 80  # 기본값 설정
        # 사업체 데이터 병합
        elif "행정구역" in biz_df.columns:
            merged_df = merged_df.merge(
                biz_df[["행정구역", biz_col]], 
                on="행정구역", 
                how="left"
            )
            # 컬럼명 통일
            merged_df.rename(columns={biz_col: "인구천명당사업체수"}, inplace=True)
    
    # 결측치 처리
    if "인구[명]" in merged_df.columns:
        merged_df["인구[명]"].fillna(10000, inplace=True)
    else:
        merged_df["인구[명]"] = 10000
        
    if "인구천명당사업체수" in merged_df.columns:
        merged_df["인구천명당사업체수"].fillna(80, inplace=True)
    else:
        merged_df["인구천명당사업체수"] = 80
    
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
        
        # 해당 대분류의 데이터만 필터링
        type_df = df[df["대분류"] == usage_type]
        
        # 권역별로 데이터 분류
        for region in type_df["권역"].dropna().unique():
            region_df = type_df[type_df["권역"] == region]
            
            if len(region_df) > 0:
                result[usage_type][region] = region_df
    
    return result


def calculate_centroids(grouped_data: Dict[str, Dict[str, pd.DataFrame]], 
                       feature_cols: List[str], 
                       method: str = "mean") -> pd.DataFrame:
    """용도 유형과 권역별 센트로이드 계산"""
    centroids = []
    
    for usage_type, regions in grouped_data.items():
        for region, region_df in regions.items():
            centroid = {"usage_type": usage_type, "region": region}
            
            # 각 특징별 센트로이드 계산
            for col in feature_cols:
                if col in region_df.columns:
                    # 계산 방법에 따라 센트로이드 계산
                    if method == "mean":
                        centroid[col] = region_df[col].mean()
                    elif method == "median":
                        centroid[col] = region_df[col].median()
                    else:
                        centroid[col] = region_df[col].mean()
            
            centroids.append(centroid)
    
    # 센트로이드 데이터프레임 생성
    centroids_df = pd.DataFrame(centroids)
    
    # 결측치 처리
    for col in feature_cols:
        if col in centroids_df.columns:
            centroids_df[col].fillna(centroids_df[col].mean(), inplace=True)
    
    return centroids_df


def preprocess_integrated_data(df: pd.DataFrame) -> pd.DataFrame:
    """통합 데이터 전처리"""
    # 컬럼명 찾기
    address_col = find_column_by_keyword(df, ["지번주소", "주소", "field6", "_CLEANADDR"])
    if not address_col:
        raise ValueError("주소 컬럼을 찾을 수 없습니다.")
    
    admin_col = find_column_by_keyword(df, ["관할주소"])
    
    # 데이터 복사 (원본 보존)
    processed_df = df.copy()
    
    # 행정구역 추출
    if admin_col and admin_col in processed_df.columns:
        processed_df["행정구역"] = processed_df[admin_col].apply(extract_admin_region)
    else:
        processed_df["행정구역"] = processed_df[address_col].apply(extract_admin_region)
    
    # 권역(도/광역시) 추출
    if admin_col and admin_col in processed_df.columns:
        processed_df["권역"] = processed_df[admin_col].apply(extract_province)
    else:
        processed_df["권역"] = processed_df[address_col].apply(extract_province)
    
    # 결측치 처리
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 컬럼명 통일
    if address_col != "주소":
        processed_df.rename(columns={address_col: "주소"}, inplace=True)
    
    # 교통량(AADT) 컬럼이 있는 경우 처리
    if "교통량(AADT)" in processed_df.columns:
        # 결측치 처리
        processed_df["교통량(AADT)"].fillna(processed_df["교통량(AADT)"].mean(), inplace=True)
    
    return processed_df