import pandas as pd
import requests
import time

ACCESS_TOKEN = "e5cf9a93-0a54-48f3-8f0a-6130f9e1c29e"

def fetch_adm_cd(lon, lat):
    url = (
        "https://sgisapi.kostat.go.kr/OpenAPI3/addr/rgeocodewgs84.json"
        f"?accessToken={ACCESS_TOKEN}&x_coor={lon}&y_coor={lat}&addr_type=21"
    )

    try:
        res = requests.get(url, timeout=3)
        js = res.json()

        # SGIS는 result 주소가 리스트로 올 때도 있음
        result = js.get("result")
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("adm_dr_cd")   # OK
        
        if isinstance(result, dict):
            return result.get("adm_dr_cd")      # OK

        return None

    except Exception as e:
        return None


df = pd.read_csv("station.csv", encoding="utf-8")

results = []
for i, row in df.iterrows():
    lon = row["_X"]   # 경도
    lat = row["_Y"]   # 위도

    adm = fetch_adm_cd(lon, lat)
    results.append(adm)

    print(f"[{i+1}/{len(df)}] → {adm}")
    time.sleep(0.05)

df["adm_cd"] = results
df.to_csv("station.csv", index=False, encoding="utf-8")

print("\n🎉 station_fixed.csv 완성됨")
