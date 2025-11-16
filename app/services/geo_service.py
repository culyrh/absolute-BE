"""
지리 정보 처리 서비스
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import re

from app.utils.data_loader import load_all_data
from app.utils.preprocessing import preprocess_gas_station_data, extract_admin_region, extract_province, normalize_region


class GeoService:
    """지리 정보 처리 서비스"""
    
    def __init__(self):
        self.data = None
        self.initialize_data()
    
    def initialize_data(self):
        """데이터 초기화 및 로드"""
        print("🚀 지리 정보 서비스 초기화 중...")
    
        self.data = {}
    
        try:
            from app.utils.data_loader import load_gas_station_data
        
            self.data["gas_station"] = load_gas_station_data()  # idx가 부여된 station 데이터
            self.data["gas_station"] = preprocess_gas_station_data(self.data["gas_station"])
        
            print(f"🔧 주유소 데이터 로드 완료: {len(self.data['gas_station'])}개 행")
            print("✅ 지리 정보 서비스 초기화 완료")
            
            gas_df = self.data["gas_station"]
            print("🔥 gas_station 컬럼:", gas_df.columns.tolist())
            print(gas_df[["위도","경도"]].head())
            print(gas_df[["위도","경도"]].dtypes)
            print("🔥 gas_station 총 개수:", len(gas_df))

        except Exception as e:
            print(f"⚠️ 지리 정보 서비스 초기화 실패: {str(e)}")
    
    def search_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """상호(주유소 이름)로 주유소 검색"""
        if not query or not self.data or "gas_station" not in self.data:
            return []
        
        gas_df = self.data["gas_station"]

        # '상호' 컬럼이 있는지 확인
        if "상호" not in gas_df.columns:
            return []
        
        # 검색어가 포함된 상호만 필터링 (대소문자 무시)
        filtered_df = gas_df[gas_df["상호"].astype(str).str.contains(query, case=False, na=False)]
        
        # 상위 limit개만 반환
        result = filtered_df.head(limit).to_dict('records')
        
        return result


    def search_by_address(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """주소로 주유소 검색"""
        if not query or not self.data or "gas_station" not in self.data:
            return []
        
        gas_df = self.data["gas_station"]
        
        # 주소 검색
        filtered_df = gas_df[gas_df["주소"].astype(str).str.contains(query, na=False)]
        
        # 상위 limit개만 반환
        result = filtered_df.head(limit).to_dict('records')
        
        return result
    
    def search_by_region(self, region: str, limit: int = 10) -> List[Dict[str, Any]]:
        """행정구역으로 주유소 검색"""
        if not region or not self.data or "gas_station" not in self.data:
            return []
        
        gas_df = self.data["gas_station"]
        
        # 행정구역 추출 (없으면 그대로 사용)
        normalized_region = normalize_region(region)
        
        # 행정구역 검색 (권역 또는 시/군/구)
        filtered_df = gas_df[
            (gas_df["행정구역"].astype(str).str.contains(region, na=False)) |
            (gas_df["권역"].astype(str).str.contains(normalized_region, na=False))
        ]
        
        # 상위 limit개만 반환
        result = filtered_df.head(limit).to_dict('records')
        
        return result
    
    def search_by_status(self, status: str, limit: int = 10) -> List[Dict[str, Any]]:
        """상태로 주유소 검색"""
        if not status or not self.data or "gas_station" not in self.data:
            return []
        
        gas_df = self.data["gas_station"]
        
        # 상태 컬럼이 있는지 확인
        if "상태" not in gas_df.columns:
            return []
        
        # 상태 검색
        filtered_df = gas_df[gas_df["상태"].astype(str).str.contains(status, na=False)]
        
        # 상위 limit개만 반환
        result = filtered_df.head(limit).to_dict('records')
        
        return result
    
    def get_all_regions(self) -> List[str]:
        """모든 권역 목록 조회"""
        if not self.data or "gas_station" not in self.data:
            return []
        
        gas_df = self.data["gas_station"]
        
        # 권역 컬럼이 있는지 확인
        if "권역" not in gas_df.columns:
            return []
        
        # 고유 권역 추출 및 정렬
        regions = sorted(gas_df["권역"].dropna().unique().tolist())
        
        return regions
    
    def get_station_by_id(self, station_id: int) -> Optional[Dict[str, Any]]:
        """ID로 주유소 조회"""
        if not self.data or "gas_station" not in self.data:
            return None
        
        gas_df = self.data["gas_station"]
        
        # ID 컬럼이 있는지 확인
        if "id" not in gas_df.columns:
            return None
        
        # ID로 검색
        filtered_df = gas_df[gas_df["id"] == station_id]
        
        if len(filtered_df) == 0:
            return None
        
        # 첫 번째 결과 반환
        return filtered_df.iloc[0].to_dict()
    
    def get_station_stats(self) -> Dict[str, Any]:
        """주유소 통계 정보"""
        if not self.data or "gas_station" not in self.data:
            return {}
        
        gas_df = self.data["gas_station"]
        
        stats = {
            "total_count": len(gas_df)
        }
        
        # 상태별 통계
        if "상태" in gas_df.columns:
            status_counts = gas_df["상태"].value_counts().to_dict()
            stats["status_counts"] = status_counts
        
        # 권역별 통계
        if "권역" in gas_df.columns:
            region_counts = gas_df["권역"].value_counts().to_dict()
            stats["region_counts"] = region_counts
        
        # 연도별 통계
        if "년도" in gas_df.columns:
            year_counts = gas_df["년도"].value_counts().to_dict()
            stats["year_counts"] = year_counts
        
        return stats
