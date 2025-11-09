"""
추천 알고리즘 기본 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd


class BaseRecommendationAlgorithm(ABC):
    """추천 알고리즘 기본 클래스"""
    
    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str]):
        """
        Args:
            centroids: 센트로이드 데이터프레임
            norm_cols: 정규화된 특징 컬럼 리스트
        """
        self.centroids = centroids
        self.norm_cols = norm_cols
    
    @abstractmethod
    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        추천 결과 생성
        
        Args:
            df: 필터링된 주유소 데이터프레임
            top_k: 반환할 결과 수
            
        Returns:
            추천 결과 리스트
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """알고리즘 이름"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """알고리즘 설명"""
        pass
    
    def _format_result(self, 
                       address_row: pd.Series, 
                       usage_type: str, 
                       score: float, 
                       rank: int,
                       **kwargs) -> Dict[str, Any]:
        """
        결과를 표준 형식으로 포맷
        
        Args:
            address_row: 주소 데이터 행
            usage_type: 용도 유형
            score: 점수
            rank: 순위
            **kwargs: 추가 정보
            
        Returns:
            포맷된 결과 딕셔너리
        """
        return {
            "address": address_row.get("주소", ""),
            "admin_region": address_row.get("행정구역", ""),
            "usage_type": usage_type,
            "score": float(score),
            "rank": rank,
            "population": float(address_row.get("인구[명]", 0)),
            "business_density": float(address_row.get("인구천명당사업체수", 0)),
            "population_norm": float(address_row.get("인구[명]_norm", 0)),
            "business_density_norm": float(address_row.get("인구천명당사업체수_norm", 0)),
            "traffic_norm": float(address_row.get("교통량_norm", 0) if "교통량_norm" in address_row else 0),
            "tourism_norm": float(address_row.get("숙박업소(관광지수)_norm", 0) if "숙박업소(관광지수)_norm" in address_row else 0),
            "land_price_norm": float(address_row.get("공시지가(토지단가)_norm", 0) if "공시지가(토지단가)_norm" in address_row else 0),
            "station_name": address_row.get("상호명", ""),
            "station_status": address_row.get("상태", ""),
            "note": address_row.get("비고", ""),
            **kwargs  # 알고리즘별 추가 정보
        }