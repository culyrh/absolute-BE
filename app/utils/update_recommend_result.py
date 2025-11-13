# 📄 app/utils/update_recommend_result.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


def update_recommend_result(train_path: str, centroid_path: str, output_path: str):
    """
    train.csv의 각 행에 대해 대분류_권역별_센터로이드.csv 기준으로
    가장 유사한 대분류를 계산해 추천결과_행단위.csv로 저장
    """
    print("🚀 추천결과_행단위.csv 갱신 시작")

    # --- 데이터 로드 ---
    train_df = pd.read_csv(train_path)
    centroid_df = pd.read_csv(centroid_path)

    # --- 공통 피처 ---
    feature_cols = ["인구[명]", "교통량(AADT)", "숙박업소(관광지수)", "상권밀집도(비율)"]
    norm_cols = [f"{col}_norm" for col in feature_cols]

    # --- train 정규화 (Z-score) ---
    for col in feature_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        if std == 0 or pd.isna(std):
            train_df[f"{col}_norm"] = 0
        else:
            train_df[f"{col}_norm"] = (train_df[col] - mean) / std

    # --- 추천 결과 저장 리스트 ---
    results = []

    # --- 각 행별로 코사인 유사도 계산 ---
    for _, row in train_df.iterrows():
        region = str(row["관할주소"]) if "관할주소" in row else None
        address_vec = np.array([row[col] for col in norm_cols]).reshape(1, -1)

        # 해당 권역의 센터로이드만 필터링
        region_centroids = centroid_df[centroid_df["권역"] == region]

        if len(region_centroids) == 0:
            # 해당 권역이 없으면 전체 중에서 탐색
            region_centroids = centroid_df.copy()

        # 코사인 유사도 계산
        centroid_vecs = region_centroids[norm_cols].fillna(0).to_numpy()
        sims = cosine_similarity(address_vec, centroid_vecs)[0]

        # 최고 유사도 대분류 선택
        best_idx = int(np.argmax(sims))
        best_usage = region_centroids.iloc[best_idx]["대분류"]
        best_sim = sims[best_idx]

        results.append({
            "대분류": row["대분류"],
            "지번주소 (읍/면/동)": row["지번주소 (읍/면/동)"],
            "관할주소": row["관할주소"],
            "인구[명]": row["인구[명]"],
            "교통량(AADT)": row["교통량(AADT)"],
            "숙박업소(관광지수)": row["숙박업소(관광지수)"],
            "상권밀집도(비율)": row["상권밀집도(비율)"],
            "추천_대분류": best_usage,
            "추천_유사도": best_sim
        })

    result_df = pd.DataFrame(results)

    # --- 저장 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 추천결과_행단위.csv 갱신 완료: {output_path}")
    print(f"총 {len(result_df)}개 행 처리 완료")


if __name__ == "__main__":
    update_recommend_result(
        train_path="data/train.csv",
        centroid_path="data/대분류_센터로이드.csv",
        output_path="data/추천결과_행단위.csv"
    )
