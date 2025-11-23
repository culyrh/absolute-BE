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
        df = pd.read_csv(file_path, dtype=str)

        # strip 해서 공백 제거
        df.columns = df.columns.str.strip()

        # -----------------------------
        # 1) 기본 컬럼 매핑
        # -----------------------------
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

        # -----------------------------
        # 2) 경도/위도 처리
        # -----------------------------
        if "경도" not in df.columns and "_X" in df.columns:
            df = df.rename(columns={"_X": "경도"})
        if "위도" not in df.columns and "_Y" in df.columns:
            df = df.rename(columns={"_Y": "위도"})

        # float 변환 시도 (실패해도 전체는 안 죽게)
        for col in ["위도", "경도"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 좌표 없는 행 제거
        if "위도" in df.columns and "경도" in df.columns:
            df = df.dropna(subset=["위도", "경도"])
        else:
            print("⚠️ 위도/경도 컬럼이 없어 주유소 위치 기반 API가 동작하지 않을 수 있습니다.")

        # -----------------------------
        # 3) 주유소만 필터
        # -----------------------------
        df = df[df["업종"] == "주유소"].copy()

        # -----------------------------
        # 4) 법정동 코드 10자리 정규화 함수
        # -----------------------------
        def normalize_code(code):
            if code is None:
                return None

            s = str(code).strip()

            if s.endswith(".0"):
                s = s[:-2]

            s = "".join(c for c in s if c.isdigit())

            if len(s) == 8:
                s = s + "00"

            if len(s) < 10:
                s = s.ljust(10, "0")

            return s[:10]

        # 법정동코드 정규화
        if "법정동코드" in df.columns:
            df["법정동코드"] = df["법정동코드"].apply(normalize_code)
        else:
            df["법정동코드"] = None

        # -----------------------------
        # 5) 법정동 전체 코드 로드 + 조인
        # -----------------------------
        bjd_path = DATA_DIR / "법정동_코드_전체자료.csv"
        if bjd_path.exists():
            try:
                df_bjd = pd.read_csv(bjd_path, dtype=str)
                df_bjd["법정동코드"] = df_bjd["법정동코드"].apply(normalize_code)

                df = df.merge(
                    df_bjd[["법정동코드", "법정동명"]],
                    how="left",
                    on="법정동코드",
                )

                # 이미 행정구역이 있으면 덮어쓰지 않음
                if "행정구역" not in df.columns:
                    df = df.rename(columns={"법정동명": "행정구역"})
            except Exception as e:
                print(f"⚠️ 법정동 코드 매핑 중 오류: {e}")
        else:
            print("⚠️ 법정동 코드 파일을 찾지 못했습니다. 행정구역 매핑 없이 진행합니다.")
            if "행정구역" not in df.columns:
                df["행정구역"] = ""

        # 결측치 제거
        df["행정구역"] = df["행정구역"].fillna("")

        
        # -----------------------------
        # 6) 인덱스 기반 ID 생성
        # -----------------------------
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