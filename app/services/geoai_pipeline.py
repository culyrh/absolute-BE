# app/services/geoai_pipeline.py

from app.services.geoai_feature_engineer import GeoAIFeatureEngineer
from app.services.geoai_model import GeoAIClassifier

class GeoAIPipeline:
    def __init__(self):
        self.engineer = GeoAIFeatureEngineer(debug=True, debug_limit=5)
        self.model = GeoAIClassifier()

    def run(self):
        df = self.engineer.run()    # 피처 엔지니어링 실행
        self.model.train(df)    # 학습 실행


if __name__ == "__main__":
    pipeline = GeoAIPipeline()
    pipeline.run()
