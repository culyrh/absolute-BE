import pandas as pd
import numpy as np
import os
from app.services.geoai_feature_engineer import GeoAIFeatureEngineer
from app.services.geoai_model import GeoAIClassifier
from app.services.geoai_config import GeoAIConfig


def clean_coord_columns(df):
    """중복 경도/위도 컬럼을 자동 정리"""

    # 1) _X, _Y → 경도, 위도
    if "_X" in df.columns:
        df["경도"] = df["_X"]
    if "_Y" in df.columns:
        df["위도"] = df["_Y"]

    # 2) 중복 컬럼 제거 (앞에 오는 경도/위도만 살리고 뒤쪽은 삭제)
    df = df.loc[:, ~df.columns.duplicated()]

    # 3) 혹시라도 공백이 있는 컬럼명 정리
    df.columns = df.columns.str.strip()

    return df


def main():

    cfg = GeoAIConfig()

    # -------------------------------
    # 1) train.csv 로드
    # -------------------------------
    print("📂 train.csv 로드")
    train_df = pd.read_csv(cfg.train_csv)

    # 좌표 중복 제거
    train_df = clean_coord_columns(train_df)

    required_train = [
        "위도", "경도", "대분류",
        "인구[명]", "교통량(AADT)",
        "숙박업소(관광지수)", "상권밀집도(비율)"
    ]

    for c in required_train:
        if c not in train_df.columns:
            raise ValueError(f"train.csv에 '{c}' 컬럼이 없습니다.")

    # -------------------------------
    # 2) train.csv 공간 피처 생성
    # -------------------------------
    print("🧭 train.csv 공간 피처 생성")
    engineer = GeoAIFeatureEngineer(debug=False)
    train_feat = engineer._compute_all_features_batch(train_df)
    train_ready = pd.concat([train_df.reset_index(drop=True), train_feat], axis=1)

    # -------------------------------
    # 3) 모델 학습
    # -------------------------------
    print("🤖 모델 학습")
    model = GeoAIClassifier()
    model.train(train_ready)

    # -------------------------------
    # 4) station.csv 로드
    # -------------------------------
    print("📂 station.csv 로드")
    station_path = cfg.station_csv
    station = pd.read_csv(station_path)

    # 🚀 여기가 핵심! 중복 경도/위도 처리
    station = clean_coord_columns(station)

    required_station = [
        "위도", "경도",
        "인구[명]", "교통량(AADT)",
        "숙박업소(관광지수)", "상권밀집도(비율)"
    ]
    for c in required_station:
        if c not in station.columns:
            raise ValueError(f"station.csv에 '{c}' 컬럼이 없습니다.")

    # -------------------------------
    # 5) station.csv 공간 피처 생성
    # -------------------------------
    print("🧮 station.csv 공간 피처 생성")
    station_feat = engineer._compute_all_features_batch(station)
    station_ready = pd.concat([station.reset_index(drop=True), station_feat], axis=1)

    # -------------------------------
    # 6) 예측 (top-3)
    # -------------------------------
    print("🔮 top-3 예측 수행")
    X_pred = station_ready[model.feature_names_]
    proba = model.clf.predict_proba(X_pred)
    classes = model.clf.classes_

    top1, top2, top3 = [], [], []

    for p in proba:
        idx = np.argsort(p)[::-1][:3]
        top1.append(classes[idx[0]])
        top2.append(classes[idx[1]])
        top3.append(classes[idx[2]])

    station["recommend1"] = top1
    station["recommend2"] = top2
    station["recommend3"] = top3

    # -------------------------------
    # 7) 덮어쓰기 저장
    # -------------------------------
    station.to_csv(station_path, index=False, encoding="utf-8-sig")
    print("station.csv에 recommend1~3 컬럼이 추가되어 저장되었습니다.")
    print("경도/위도 중복도 자동 정리 완료!")


if __name__ == "__main__":
    main()
