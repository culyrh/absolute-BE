"""
데이터 로드 유틸리티
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from app.core.config import settings, DATA_DIR


def load_gas_station_data() -> pd.DataFrame:
    """주유소 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.GAS_STATION_FILE   # "jeonju_gas_station.csv"
        df = pd.read_csv(file_path)
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


def load_closed_gas_station_data() -> pd.DataFrame:
    """폐/휴업 주유소 데이터 로드"""
    try:
        file_path = DATA_DIR / settings.CLOSED_GAS_STATION_FILE   # "폐주유소좌표변환.csv"
        df = pd.read_csv(file_path)
        
        # 컬럼 이름 매핑 - 새로운 CSV 파일 형식에 맞게 조정
        column_mapping = {
            "field1": "년도", 
            "field2": "일자",
            "field3": "업종",
            "field4": "상태",
            "field5": "상호",
            "field6": "주소",
            "_CLEANADDR": "정제주소",
            "_X": "경도",
            "_Y": "위도"
        }
        
        # 필요한 컬럼이 있는지 확인하고 이름 변경
        df = df.rename(columns=column_mapping)
        
        # 필터링: 폐업 또는 휴업 상태인 레코드만 선택
        df = df[df["상태"].isin(["폐업", "휴업"])]
        
        print(f"📊 폐/휴업 주유소 데이터 로드 완료: {len(df)}개 행")
        return df
    except Exception as e:
        print(f"❌ 폐/휴업 주유소 데이터 로드 실패: {str(e)}")
        # 오류 발생 시 빈 DataFrame 반환 - 애플리케이션이 중단되지 않도록 함
        return pd.DataFrame(columns=["년도", "일자", "업종", "상태", "상호", "주소", "정제주소", "경도", "위도"])


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
            "closed_gas_station": load_closed_gas_station_data()
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
        
        # 폐/휴업 주유소 데이터는 항상 빈 DataFrame이라도 제공
        data["closed_gas_station"] = load_closed_gas_station_data()
    
    print("✅ 모든 데이터 로드 완료")
    return data