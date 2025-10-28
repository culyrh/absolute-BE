"""
데이터 로드 유틸리티
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from app.core.config import settings, DATA_DIR


def load_gas_station_data() -> pd.DataFrame:
    """주유소 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.GAS_STATION_FILE
        df = pd.read_csv(file_path)
        print(f"📊 주유소 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 주유소 데이터 로드 실패: {str(e)}")
        raise


def load_population_data() -> pd.DataFrame:
    """인구수 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.POPULATION_FILE
        df = pd.read_excel(file_path)
        print(f"📊 인구수 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 인구수 데이터 로드 실패: {str(e)}")
        raise


def load_business_data() -> pd.DataFrame:
    """사업체 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.BUSINESS_FILE
        df = pd.read_excel(file_path)
        print(f"📊 사업체 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 사업체 데이터 로드 실패: {str(e)}")
        raise


def load_centroid_data() -> pd.DataFrame:
    """센트로이드 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.CENTER_FILE
        df = pd.read_csv(file_path)
        print(f"📊 센트로이드 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 센트로이드 데이터 로드 실패: {str(e)}")
        raise


def load_recommend_result_data() -> pd.DataFrame:
    """추천 결과 행단위 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.RECOMMEND_RESULT_FILE
        df = pd.read_csv(file_path)
        print(f"📊 추천 결과 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 추천 결과 데이터 로드 실패: {str(e)}")
        raise


def load_closed_gas_station_data() -> pd.DataFrame:
    """폐/휴업 주유소 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.CLOSED_GAS_STATION_FILE
        df = pd.read_csv(file_path)
        print(f"📊 폐/휴업 주유소 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 폐/휴업 주유소 데이터 로드 실패: {str(e)}")
        raise


def find_column_by_keyword(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """키워드를 포함하는 컬럼명 찾기"""
    for keyword in keywords:
        for column in df.columns:
            if keyword in column:
                return column
    return None


def load_all_data() -> Dict[str, pd.DataFrame]:
    """모든 필요 데이터 로드"""
    print("📂 전체 데이터 로드 시작...")
    
    data = {
        "gas_station": load_gas_station_data(),
        "population": load_population_data(),
        "business": load_business_data(),
        "centroid": load_centroid_data(),
        "recommend_result": load_recommend_result_data(),
        "closed_gas_station": load_closed_gas_station_data()
    }
    
    print("✅ 모든 데이터 로드 완료")
    return data
