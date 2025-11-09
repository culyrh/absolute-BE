"""
ML 기반 대분류 추천 서비스
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MLLocationRecommender:
    DATA_DIR_NAME = "data"
    TRAIN_FILE_NAME = "데이터 통합 - 시트1.csv"
    STATION_FILE_NAME = "gas_station_features.csv"

    FEATURE_COLS = ["인구[명]", "교통량(AADT)", "숙박업소(관광지수)", "상권밀집도(비율)"]
    TARGET_COL = "대분류"

    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.base_dir / self.DATA_DIR_NAME

        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        self.station_df: Optional[pd.DataFrame] = None

    # ========== 내부 유틸 ==========

    def _load_train_df(self) -> pd.DataFrame:
        path = self.data_dir / self.TRAIN_FILE_NAME
        df = pd.read_csv(path, encoding="utf-8-sig")

        for col in self.FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=self.FEATURE_COLS + [self.TARGET_COL])
        return df

    def _build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, self.FEATURE_COLS)]
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )

        pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", model),
            ]
        )
        return pipeline

    def _ensure_trained(self) -> None:
        if self.pipeline is None or self.classes_ is None:
            raise RuntimeError("MLLocationRecommender: train()을 먼저 호출해야 합니다.")

    def _ensure_station_df(self) -> None:
        if self.station_df is not None:
            return

        path = self.data_dir / self.STATION_FILE_NAME

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            df = pd.read_excel(path)

        self.station_df = df

    # ========== 학습 ==========

    def train(self) -> float:
        df = self._load_train_df()

        X = df[self.FEATURE_COLS]
        y = df[self.TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        pipeline = self._build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[MLLocationRecommender] accuracy = {acc:.3f}")
        try:
            print(classification_report(y_test, y_pred))
        except Exception:
            pass

        self.pipeline = pipeline
        self.classes_ = pipeline.named_steps["model"].classes_
        print("[MLLocationRecommender] 학습된 대분류 클래스:", list(self.classes_))
        return acc

    # ========== 예측 (row → top-N) ==========

    def _predict_from_row(
        self,
        row: pd.Series,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        self._ensure_trained()

        sample = row[self.FEATURE_COLS].to_frame().T.copy()
        for col in self.FEATURE_COLS:
            sample[col] = pd.to_numeric(sample[col], errors="coerce")
        sample = sample.fillna(0)

        proba = self.pipeline.predict_proba(sample)[0]
        indices = np.argsort(proba)[::-1][:top_n]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "category": str(self.classes_[idx]),
                    "probability": float(round(proba[idx], 4)),
                }
            )
        return results

    # ========== 외부에서 쓰는 메인 함수 ==========

    def recommend_for_station(
        self,
        keyword: str,
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """
        상호명/업체명/주소 등에 keyword가 포함된 주유소를 찾아
        그 행의 피처로 대분류 top-N 추천.
        """
        self._ensure_trained()
        self._ensure_station_df()

        df = self.station_df

        # 1) 실제 존재하는 이름/주소 컬럼 자동 감지
        name_candidates = ["상호명", "업체명", "상호", "주유소명"]
        addr_candidates = ["주소", "소재지", "관할주소"]

        name_cols = [c for c in name_candidates if c in df.columns]
        addr_cols = [c for c in addr_candidates if c in df.columns]

        if not name_cols and not addr_cols:
            raise ValueError("gas_station_features에서 주유소 이름/주소 컬럼을 찾을 수 없습니다.")

        # 2) 검색 마스크 생성
        masks = []

        for c in name_cols:
            masks.append(df[c].astype(str).str.contains(keyword, na=False))

        for c in addr_cols:
            masks.append(df[c].astype(str).str.contains(keyword, na=False))

        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m

        candidates = df[mask]

        if candidates.empty:
            return {
                "keyword": keyword,
                "matched": False,
                "station": None,
                "results": [],
                "message": f"'{keyword}' 에 해당하는 주유소를 gas_station_features에서 찾을 수 없습니다.",
            }

        # 일단 첫 번째 매칭만 사용
        row = candidates.iloc[0]

        results = self._predict_from_row(row, top_n=top_n)

        # 대표 이름/주소 컬럼 하나씩 선택
        name_col = name_cols[0] if name_cols else None
        addr_col = addr_cols[0] if addr_cols else None

        station_info = {
            "년도": int(row.get("년도")) if "년도" in row and not pd.isna(row.get("년도")) else None,
            "날짜": row.get("날짜"),
            "분류": row.get("분류"),
            "상태": row.get("상태"),
            "업체명": row.get(name_col) if name_col else None,
            "주소": row.get(addr_col) if addr_col else None,
            "잔존여부": row.get("잔존여부") if "잔존여부" in row else None,
            "비고": row.get("비고") if "비고" in row else None,
            "관할주소": row.get("관할주소") if "관할주소" in row else None,
            "인구[명]": float(row.get("인구[명]")) if "인구[명]" in row and not pd.isna(row.get("인구[명]")) else None,
            "교통량(AADT)": float(row.get("교통량(AADT)")) if "교통량(AADT)" in row and not pd.isna(row.get("교통량(AADT)")) else None,
            "숙박업소(관광지수)": float(row.get("숙박업소(관광지수)")) if "숙박업소(관광지수)" in row and not pd.isna(row.get("숙박업소(관광지수)")) else None,
            "상권밀집도(비율)": float(row.get("상권밀집도(비율)")) if "상권밀집도(비율)" in row and not pd.isna(row.get("상권밀집도(비율)")) else None,
        }

        return {
            "keyword": keyword,
            "matched": True,
            "station": station_info,
            "results": results,
            "message": f"{len(candidates)}개 주유소 중 1개를 사용해 추천을 수행했습니다.",
        }


if __name__ == "__main__":
    rec = MLLocationRecommender()
    rec.train()
    print(rec.recommend_for_station("상무제일주유소", top_n=3))
