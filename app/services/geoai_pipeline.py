# app/services/geoai_pipeline.py

from app.services.geoai_feature_engineer import GeoAIFeatureEngineer
from app.services.geoai_model import GeoAIClassifier

class GeoAIPipeline:
    def __init__(self):
        self.engineer = GeoAIFeatureEngineer()
        self.model = GeoAIClassifier()

    def run(self):
        df = self.engineer.run()
        self.model.train(df)


if __name__ == "__main__":
    pipeline = GeoAIPipeline()
    pipeline.run()
