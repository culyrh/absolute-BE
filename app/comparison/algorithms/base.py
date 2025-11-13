import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BaseAlgorithm:
    """추천 알고리즘 기본 클래스"""

    def __init__(self, centroids: pd.DataFrame, norm_cols: list, train_data: pd.DataFrame = None):
        self.centroids = centroids
        self.norm_cols = norm_cols
        self.train_data = train_data

    def _normalize_if_missing(self, df: pd.DataFrame):
        """_norm 컬럼이 누락된 경우 자동 정규화"""
        missing = [c for c in self.norm_cols if c not in df.columns]
        if missing:
            scaler = StandardScaler()
            normed = scaler.fit_transform(df[[c.replace("_norm", "") for c in self.norm_cols if c.replace("_norm", "") in df.columns]])
            for i, col in enumerate(self.norm_cols):
                if col.replace("_norm", "") in df.columns:
                    df[col] = normed[:, i]
        return df

    def _filter_by_region(self, df: pd.DataFrame, region_value: str):
        """권역 필터링 (없으면 전체 사용)"""
        if "권역" in df.columns:
            region_df = df[df["권역"] == region_value]
            if len(region_df) > 0:
                return region_df
        if "관할주소" in df.columns:
            region_df = df[df["관할주소"] == region_value]
            if len(region_df) > 0:
                return region_df
        return df  # fallback

    def _extract_usage_name(self, row: pd.Series):
        """usage_type 또는 대분류 이름 추출"""
        return row.get("usage_type", row.get("대분류", ""))
