"""
AHP-TOPSIS 기반 추천 알고리즘 (권역 필터 & 컬럼 호환 개선판)
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np

from app.comparison.algorithms.base import BaseAlgorithm as BaseRecommendationAlgorithm


class AHPTopsisAlgorithm(BaseRecommendationAlgorithm):
    """AHP-TOPSIS 기반 추천"""

    def __init__(self, centroids: pd.DataFrame, norm_cols: List[str], train_data: pd.DataFrame):
        """
        Args:
            centroids: 센트로이드 데이터프레임 (미사용이지만 인터페이스 통일을 위해 유지)
            norm_cols: 정규화된 특징 컬럼 리스트
            train_data: 학습 데이터 (추천결과_행단위.csv 혹은 train 기반 특징 포함 테이블)
        """
        super().__init__(centroids, norm_cols)
        self.train_data = train_data

        # AHP 가중치 정의 (고정값)
        # ※ norm_cols와 교집합만 사용 (테스트/데이터에 없는 컬럼 자동 제외)
        self.weights = {
            "인구[명]_norm": 0.30,
            "교통량_norm": 0.25,
            "숙박업소(관광지수)_norm": 0.20,
            "상권밀집도(비율)_norm": 0.25,
        }

    @property
    def name(self) -> str:
        return "ahp_topsis"

    @property
    def description(self) -> str:
        return "AHP 가중치와 TOPSIS 다기준 의사결정으로 최적 용도를 추천합니다."

    def _extract_region(self, row: pd.Series) -> str:
        """관할주소/권역/주소 텍스트에서 권역 후보 추출"""
        region = str(row.get("관할주소", "") or row.get("권역", "")).strip()
        if region:
            return region

        # 주소 텍스트에서 지역 키워드 힌트 (fallback)
        addr = str(row.get("주소", "") or row.get("지번주소 (읍/면/동)", ""))
        # 넓은 매칭 대신, 실제 권역 명칭이 그대로 들어오면 더 정확함
        for r in [
            "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시",
            "경기도", "강원특별자치도", "충청북도", "충청남도", "전북특별자치도", "전라남도", "경상북도", "경상남도",
            "제주특별자치도",
        ]:
            if r in addr:
                return r
        return ""

    def _available_cols(self, row: pd.Series) -> List[str]:
        """weights ∩ norm_cols ∩ (row, train_data 공통)"""
        # row에 있는 norm 컬럼
        row_cols = set([c for c in row.index if c.endswith("_norm")])
        # train_data에 있는 norm 컬럼
        train_cols = set([c for c in getattr(self.train_data, "columns", []) if c.endswith("_norm")])
        # weights와 교집합
        weight_cols = set(self.weights.keys())
        # 최종 사용 컬럼
        return list((row_cols & train_cols & weight_cols))

    def recommend(self, df: pd.DataFrame, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        AHP-TOPSIS 기반 추천
          1) (대분류, 권역)별 데이터에서 지표 중앙값으로 '대안' 벡터 구성
          2) 입력 지점과 대안 간 유사도 → 가중치 적용
          3) TOPSIS 상대근접도(closeness)로 순위화
        """
        try:
            if len(df) == 0 or self.train_data is None or len(self.train_data) == 0:
                return []

            # 첫 행만 사용 (현재 인터페이스 기준)
            address_row = df.iloc[0].copy()

            # 사용 가능한 norm 컬럼 결정
            available_cols = self._available_cols(address_row)
            if not available_cols:
                return []

            # 권역 추출 및 권역별 필터링 (없으면 전체 사용)
            region = self._extract_region(address_row)
            region_df = self.train_data
            if region and ("권역" in self.train_data.columns):
                sub = self.train_data[self.train_data["권역"] == region]
                if len(sub) > 0:
                    region_df = sub

            # 대분류 컬럼 확인
            if "대분류" not in region_df.columns:
                return []

            usage_types = region_df["대분류"].dropna().unique().tolist()
            if not usage_types:
                return []

            # ---------- 1) (권역 내) 대분류별 '결정 행렬' : 중앙값 ----------
            decision_matrix: Dict[str, Dict[str, float]] = {}
            for usage in usage_types:
                tdf = region_df[region_df["대분류"] == usage]
                if len(tdf) == 0:
                    # 데이터 없으면 0.5로 중립값 설정
                    decision_matrix[usage] = {c: 0.5 for c in available_cols}
                else:
                    med = {}
                    for c in available_cols:
                        if c in tdf.columns:
                            m = float(tdf[c].median())
                            # NaN 방지
                            med[c] = 0.0 if np.isnan(m) else m
                        else:
                            med[c] = 0.5
                    decision_matrix[usage] = med

            # ---------- 2) 입력 지점의 지표값 ----------
            site_values = {c: float(address_row.get(c, 0.0)) for c in available_cols}

            # ---------- 3) 유사도 행렬 ----------
            #   주의: z-score 정규화인 경우 |site - med| 가 1보다 커질 수 있으므로
            #   안전한 유사도 변환: sim = 1 / (1 + |diff|)
            similarity_matrix: Dict[str, Dict[str, float]] = {}
            for usage, medians in decision_matrix.items():
                sims = {}
                for c in available_cols:
                    diff = abs(site_values[c] - medians[c])
                    sims[c] = 1.0 / (1.0 + diff)
                similarity_matrix[usage] = sims

            # ---------- 4) AHP 가중치 적용 ----------
            weighted_matrix: Dict[str, Dict[str, float]] = {}
            for usage, sims in similarity_matrix.items():
                weighted = {}
                for c in available_cols:
                    w = self.weights.get(c, 0.0)
                    weighted[c] = w * sims[c]
                weighted_matrix[usage] = weighted

            # ---------- 5) 이상해/반대해 ----------
            ideal_positive: Dict[str, float] = {}
            ideal_negative: Dict[str, float] = {}
            for c in available_cols:
                vals = [weighted_matrix[u][c] for u in weighted_matrix]
                ideal_positive[c] = max(vals) if vals else 0.0
                ideal_negative[c] = min(vals) if vals else 0.0

            # ---------- 6) 이상해/반대해와의 거리 ----------
            dist_pos: Dict[str, float] = {}
            dist_neg: Dict[str, float] = {}
            for usage, weighted in weighted_matrix.items():
                dp = sum((weighted[c] - ideal_positive[c]) ** 2 for c in available_cols) ** 0.5
                dn = sum((weighted[c] - ideal_negative[c]) ** 2 for c in available_cols) ** 0.5
                dist_pos[usage] = dp
                dist_neg[usage] = dn

            # ---------- 7) 상대근접도 ----------
            closeness: Dict[str, float] = {}
            for usage in weighted_matrix.keys():
                dp, dn = dist_pos[usage], dist_neg[usage]
                closeness[usage] = 0.0 if (dp + dn) == 0 else float(dn / (dp + dn))

            # ---------- 8) 정렬 & top_k ----------
            sorted_results = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:top_k]

            # ---------- 9) 결과 포맷 ----------
            recommendations: List[Dict[str, Any]] = []
            for i, (usage_type, score) in enumerate(top_results):
                recommendations.append(
                    self._format_result(
                        address_row=address_row,
                        usage_type=usage_type,        # ← '대분류' 문자열
                        score=float(score),
                        rank=i + 1,
                        topsis_score=float(score),
                        ahp_weights={c: self.weights[c] for c in available_cols},
                        region=region,
                    )
                )

            return recommendations

        except Exception as e:
            print(f"⚠️ AHP-TOPSIS 추천 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
