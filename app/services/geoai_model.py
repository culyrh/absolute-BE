# app/services/geoai_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class GeoAIClassifier:
    def __init__(self):
        self.clf = None
        self.feature_names_ = []

    def train(self, df: pd.DataFrame):
        feature_cols = [
            "인구[명]",
            "교통량(AADT)",
            "숙박업소(관광지수)",
            "상권밀집도(비율)",

            "parcel_300m",
            "parcel_500m",
            "nearest_parcel_m",

            "poi_store_300m",
            "poi_hotel_300m",
            "poi_restaurant_300m",
        ]

        self.feature_names_ = feature_cols

        X = df[feature_cols]
        y = df["대분류"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        print("📊 === train 내부 검증 성능 ===")
        print(classification_report(y_test, preds))

        self.clf = clf
        return clf
