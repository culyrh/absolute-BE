# app/services/geoai_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class GeoAIClassifier:
    def train(self, df: pd.DataFrame):
        X = df[["인구[명]", "교통량(AADT)", "숙박업소(관광지수)", "상권밀집도(비율)", 
                "parcel_300m", "parcel_500m"]]
        y = df["대분류"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

        clf = RandomForestClassifier(n_estimators=200)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))

        return clf
