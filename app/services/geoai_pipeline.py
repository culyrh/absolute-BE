# app/services/geoai_pipeline.py

from app.services.geoai_feature_engineer import GeoAIFeatureEngineer
from app.services.geoai_model import GeoAIClassifier

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import os
import numpy as np
import folium
from collections import Counter
from sklearn.metrics import classification_report, precision_score, recall_score


class GeoAIPipeline:
    def __init__(self):
        self.engineer = GeoAIFeatureEngineer()
        self.model = GeoAIClassifier()

    # --------------------- Train ---------------------
    def run(self):
        df = self.engineer.run()
        clf = self.model.train(df)
        self.model.clf = clf

    # ---------------- Feature Importance ----------------
    def plot_feature_importance(self):
        clf = self.model.clf
        feature_names = self.model.feature_names_

        importances = clf.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(8, 6))
        plt.title("GeoAI Feature Importance")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    # ---------------- Precision / Recall Graph ----------------
    def plot_precision_recall(self, y_true, y_pred):
        classes = sorted(list(set(y_true)))

        precisions = precision_score(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )
        recalls = recall_score(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )

        x = np.arange(len(classes))
        width = 0.35

        plt.figure(figsize=(12, 7))
        plt.bar(x - width/2, precisions, width, label="Precision")
        plt.bar(x + width/2, recalls, width, label="Recall")

        plt.xlabel("대분류")
        plt.ylabel("Score")
        plt.title("Precision / Recall per Class")
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------------- Visualization (HTML Map + Buffer) -----------------
    def visualize_on_map(self, df):
        m = folium.Map(
            location=[df["위도"].mean(), df["경도"].mean()],
            zoom_start=11
        )

        # 버퍼 레이어와 포인트 레이어 분리
        buf300_layer = folium.FeatureGroup(name="300m Buffer", show=False)
        buf500_layer = folium.FeatureGroup(name="500m Buffer", show=False)
        point_layer = folium.FeatureGroup(name="Stations", show=True)

        for _, row in df.iterrows():
            lat = float(row["위도"])
            lon = float(row["경도"])

            # 300m / 500m 버퍼 (원)
            folium.Circle(
                location=[lat, lon],
                radius=300,
                color="blue",
                fill=False,
                weight=1,
                opacity=0.5
            ).add_to(buf300_layer)

            folium.Circle(
                location=[lat, lon],
                radius=500,
                color="green",
                fill=False,
                weight=1,
                opacity=0.5
            ).add_to(buf500_layer)

            # 포인트
            html = f"""
            <hr>
            <b>parcel_300m:</b> {row['parcel_300m']}<br>
            <b>parcel_500m:</b> {row['parcel_500m']}<br>
            <b>nearest_parcel_m:</b> {row['nearest_parcel_m']:.2f}m<br>
            <hr>
            <b>POI 300m</b><br>
            편의점: {row['poi_store_300m']}<br>
            숙박시설: {row['poi_hotel_300m']}<br>
            음식점: {row['poi_restaurant_300m']}<br>
            """

            folium.CircleMarker(
                [lat, lon],
                radius=5,
                fill=True,
                color="red",
                fill_color="blue",
                popup=folium.Popup(html, max_width=300)
            ).add_to(point_layer)

        buf300_layer.add_to(m)
        buf500_layer.add_to(m)
        point_layer.add_to(m)

        folium.LayerControl().add_to(m)

        # 같은 services 폴더에 저장
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(CURRENT_DIR, "test_map.html")
        m.save(output_path)
        print("📌 test_map.html 생성됨")

    # ---------------------- Test ---------------------
    def evaluate_on_test(self, path):
        test_df = self.engineer.run_test(path)

        X_test = test_df[self.model.feature_names_]
        preds = self.model.clf.predict(X_test)

        if "대분류" in test_df.columns:
            print("📊 === test 성능 ===")
            print(classification_report(test_df["대분류"], preds))

            # 클래스별 개수 디버깅 (근린생활시설 안 나오는지 확인용)
            print("🔎 y_true class counts:", Counter(test_df["대분류"]))
            print("🔎 y_pred class counts:", Counter(preds))

            # 그래프
            self.plot_feature_importance()
            self.plot_precision_recall(test_df["대분류"], preds)

        test_df["예측대분류"] = preds

        self.visualize_on_map(test_df)

        return test_df


if __name__ == "__main__":
    pipe = GeoAIPipeline()
    pipe.run()
    pipe.evaluate_on_test(r"data/test_data.csv")
