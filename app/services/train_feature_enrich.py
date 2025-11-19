import pandas as pd
from app.services.geoai_feature_engineer import GeoAIFeatureEngineer
from app.services.geoai_config import GeoAIConfig

def main():
    cfg = GeoAIConfig()

    print("📂 train.csv 로드 중...")
    train_df = pd.read_csv(cfg.train_csv)

    # 위도/경도가 이미 있으므로 바로 FeatureEngineer 적용 가능
    engineer = GeoAIFeatureEngineer(debug=True)

    print("🧮 train.csv 공간 피처 생성 중...")
    enriched = engineer.run()   # train.csv 전용 엔지니어링

    print("💾 저장 중 → data/train_with_parcel.csv")
    enriched.to_csv(cfg.data_dir / "train.csv", index=False)

    print("🎉 완료: train_with_parcel.csv 생성됨")

if __name__ == "__main__":
    main()
