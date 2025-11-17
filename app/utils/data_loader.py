"""
데이터 로드 유틸리티
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from app.core.config import get_settings, DATA_DIR
settings = get_settings()

def load_gas_station_data() -> pd.DataFrame:
    try:
        file_path = settings.GAS_STATION_FILE
        df = pd.read_csv(file_path)

        # strip 해서 공백 제거
        df.columns = df.columns.str.strip()

        # 1) 기본 매핑
        column_mapping = {
            "field1": "년도", 
            "field2": "일자",
            "field3": "업종",
            "field4": "상태",
            "field5": "상호",
            "field6": "주소",
            "_GC_TYPE": "지번종류",
            "_CLEANADDR": "정제주소",
            "_PNU": "PNU",
            "숙박업소(관광지수)": "관광지수",
            "인구[명]": "인구",
            "상권밀집도(비율)": "상권밀집도",
            "교통량(AADT)": "교통량",
            "adm_cd2": "법정동코드"
        }
        df = df.rename(columns=column_mapping)

        # 2) 경도/위도 중복 생성을 방지
        # station.csv 원본에 이미 경도/위도가 있다면 rename하지 않는다.
        if "경도" not in df.columns and "_X" in df.columns:
            df = df.rename(columns={"_X": "경도"})
        if "위도" not in df.columns and "_Y" in df.columns:
            df = df.rename(columns={"_Y": "위도"})

        # 3) 주유소만 필터
        df = df[df["업종"] == "주유소"].copy()

        # 4) ID 부여
        df = df.reset_index(drop=True)
        df["id"] = df.index

        print(f"📊 주유소 데이터 로드 완료: {len(df)}개 행")
        return df

    except Exception as e:
        print(f"❌ 주유소 데이터 로드 실패: {str(e)}")
        raise


def load_population_data() -> pd.DataFrame:
    """인구수 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.POPULATION_FILE   # "전국인구수_행정동별.csv"
        df = pd.read_csv(file_path)
        print(f"📊 인구수 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 인구수 데이터 로드 실패: {str(e)}")
        raise


def load_business_data() -> pd.DataFrame:
    """사업체 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.BUSINESS_FILE   # "전국1000명당사업체수_행정동별.csv"
        df = pd.read_csv(file_path)
        print(f"📊 사업체 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 사업체 데이터 로드 실패: {str(e)}")
        raise


def load_centroid_data() -> pd.DataFrame:
    """센트로이드 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.CENTER_FILE   # "대분류_센터로이드.csv"
        df = pd.read_csv(file_path)
        print(f"📊 센트로이드 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 센트로이드 데이터 로드 실패: {str(e)}")
        raise


def load_recommend_result_data() -> pd.DataFrame:
    """추천 결과 행단위 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.RECOMMEND_RESULT_FILE   # "추천결과_행단위.csv"
        df = pd.read_csv(file_path)
        print(f"📊 추천 결과 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 추천 결과 데이터 로드 실패: {str(e)}")
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
    
    try:
        data = {
            "gas_station": load_gas_station_data(),
            "population": load_population_data(),
            "business": load_business_data(),
            "centroid": load_centroid_data(),
            "recommend_result": load_recommend_result_data(),
        }
    except Exception as e:
        print(f"⚠️ 일부 데이터 로드 실패: {str(e)}")
        # 필수 데이터만 로드하도록 재시도
        data = {}
        
        # 필수 데이터 로드 시도
        try:
            data["gas_station"] = load_gas_station_data()
        except:
            data["gas_station"] = pd.DataFrame()
        
        try:
            data["population"] = load_population_data()
        except:
            data["population"] = pd.DataFrame()
        
        try:
            data["business"] = load_business_data()
        except:
            data["business"] = pd.DataFrame()
        
        try:
            data["centroid"] = load_centroid_data()
        except:
            data["centroid"] = pd.DataFrame()
        
        try:
            data["recommend_result"] = load_recommend_result_data()
        except:
            data["recommend_result"] = pd.DataFrame()
        
    
    print("✅ 모든 데이터 로드 완료")
    return data