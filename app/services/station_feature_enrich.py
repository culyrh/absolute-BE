# app/services/station_feature_enrich.py

import pandas as pd
from app.services.geoai_feature_engineer import GeoAIFeatureEngineer

# 1. station.csv 불러오기
df = pd.read_csv("data/station.csv")

# 2. GeoAI 엔지니어러가 요구하는 형식으로 좌표 컬럼 생성
df["위도"] = df["_Y"]
df["경도"] = df["_X"]

# 3. 필요한 최소 컬럼만 복사해서 GeoAI로 계산
engineer = GeoAIFeatureEngineer(debug=True)

df_points = df[["위도", "경도"]].copy()

# 4. Feature Engineer 실행
feat = engineer._compute_all_features_batch(df_points)

# 5. 결과 병합 (parcel_300m, parcel_500m만 사용)
df["parcel_300m"] = feat["parcel_300m"]
df["parcel_500m"] = feat["parcel_500m"]

# 6. 저장
df.to_csv("data/station.csv", index=False)
print("🎉 완료됨: station_with_parcel.csv 생성됨")
