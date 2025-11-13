# app/utils/update_centroids_by_region.py

import pandas as pd
import os

def update_centroids_by_region(train_path: str, output_path: str):
    """
    최신 train.csv 기준으로
    (대분류, 권역)별 정규화 평균값(센터로이드) 갱신
    """
    df = pd.read_csv(train_path)

    # --- 정규화할 피처 목록 ---
    feature_cols = ["인구[명]", "교통량(AADT)", "숙박업소(관광지수)", "상권밀집도(비율)"]
    norm_cols = [f"{col}_norm" for col in feature_cols]

    # --- 결측치 제거 ---
    df = df.dropna(subset=feature_cols, how="all")

    # --- 피처 정규화 (Z-score) ---
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df[f"{col}_norm"] = 0
        else:
            df[f"{col}_norm"] = (df[col] - mean) / std

    # --- 권역 컬럼 정리 (관할주소 → 권역) ---
    # train.csv에서 '관할주소' 컬럼이 '강원특별자치도' 등 권역 역할을 함
    if "관할주소" in df.columns:
        df["권역"] = df["관할주소"].astype(str)
    elif "region" in df.columns:
        df["권역"] = df["region"].astype(str)
    else:
        raise ValueError("train.csv에 '관할주소' 또는 'region' 컬럼이 필요합니다.")

    # --- (대분류, 권역)별 평균 계산 ---
    centroids = (
        df.groupby(["대분류", "권역"])[norm_cols]
        .mean()
        .reset_index()
    )

    # --- 저장 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    centroids.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ (대분류 × 권역)별 센터로이드 갱신 완료: {output_path}")
    print(f"총 {len(centroids)}개 행 생성 (예상 144개)")

if __name__ == "__main__":
    update_centroids_by_region("data/train.csv", "data/대분류_센터로이드.csv")
