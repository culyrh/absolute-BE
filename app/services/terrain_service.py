# app/services/terrain_service.py
import os
import io
import json
import requests
import psycopg2
from typing import List

from PIL import Image, ImageDraw
from shapely import wkb
from shapely.geometry import Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from psycopg2.extras import RealDictCursor

from app.services.terrain_utils import lonlat_to_webmerc

VWORLD_API_KEY = os.getenv("VWORLD_API_KEY")

# VWorld WMS DEM/Hillshade
VWORLD_WMS = (
    "https://xdworld.vworld.kr/xdworld/wms?"
    "SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
    "&LAYERS=dem"
    "&STYLES=hillshade"
    "&FORMAT=image/png"
    "&SRS=EPSG:3857"
    "&WIDTH={width}&HEIGHT={height}"
    "&BBOX={minx},{miny},{maxx},{maxy}"
    "&apiKey={apiKey}"
)


class TerrainMapService:
    """VWorld Hillshade + PostGIS parcels → PNG / HTML"""

    def __init__(self, pg_dsn: str):
        self.pg_dsn = pg_dsn
        # parcels(5186) → 타일(3857) 변환
        self.tr_5186_to_3857 = Transformer.from_crs(5186, 3857, always_xy=True)

    # ----------------------------
    # 1) 중심점 기준 bbox 계산
    # ----------------------------
    def compute_bbox_around(self, lon: float, lat: float, meter: int = 500):
        cx, cy = lonlat_to_webmerc(lon, lat)
        return (cx - meter, cy - meter, cx + meter, cy + meter)

    # ----------------------------
    # 2) VWorld WMS hillshade 가져오기
    # ----------------------------
    def fetch_hillshade(self, bbox, width=768, height=768):
        minx, miny, maxx, maxy = bbox

        url = VWORLD_WMS.format(
            width=width,
            height=height,
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            apiKey=VWORLD_API_KEY or ""
        )

        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
        except Exception as e:
            print(f"[HILLSHADE ERROR] {e}")
            return Image.new("RGBA", (width, height), (160, 160, 160, 255))

    # ----------------------------
    # 3) PostGIS parcels + 속성 가져오기
    #    (지번 + 용도지역 zoning_* 다 가져옴)
    # ----------------------------
    def query_parcels(self, lon, lat, radius=500):
        conn = psycopg2.connect(self.pg_dsn)
        try:
            with conn.cursor() as cur:
                sql = """
                    SELECT
                        ST_AsBinary(geom) AS geom,
                        pnu,
                        jibun,
                        zoning_lclass,
                        zoning_mclass,
                        zoning_sclass,
                        zoning_name,
                        zoning_area,
                        zoning_notice_date
                    FROM parcels
                    WHERE ST_DWithin(
                        ST_SetSRID(geom, 5186),
                        ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 5186),
                        %s
                    )
                """
                cur.execute(sql, (lon, lat, radius))
                rows = cur.fetchall()

                result = []
                for (
                    geom_bytes,
                    pnu,
                    jibun,
                    z_l,
                    z_m,
                    z_s,
                    z_n,
                    z_area,
                    z_date
                ) in rows:
                    result.append({
                        "geom": bytes(geom_bytes),
                        "pnu": pnu,
                        "jibun": jibun,
                        "zoning_lclass": z_l,
                        "zoning_mclass": z_m,
                        "zoning_sclass": z_s,
                        "zoning_name": z_n,
                        "zoning_area": z_area,
                        "zoning_notice_date": z_date,
                    })
                return result
        finally:
            conn.close()

    # ----------------------------
    # 4) PNG 오버레이 (용도지역 색상 + 3D + 버퍼)
    # ----------------------------
    def draw_overlay(self, base_img, bbox, lon, lat, parcels):
        minx, miny, maxx, maxy = bbox
        draw = ImageDraw.Draw(base_img, "RGBA")

        # 3857 → 픽셀 좌표
        def proj_3857_to_px(x, y):
            return (
                int((x - minx) / (maxx - minx) * base_img.width),
                int((maxy - y) / (maxy - miny) * base_img.height),
            )

        # 용도지역 → 색상 매핑
        def pick_color(zoning_name, zoning_lclass):
            key = (zoning_name or zoning_lclass or "").strip()

            if not key:
                return (230, 230, 230, 60)

            # 대분류 기반
            if "상업" in key:
                return (255, 220, 190, 90)   # 상업지역
            if "주거" in key:
                return (220, 235, 255, 90)   # 주거지역
            if "공업" in key:
                return (255, 210, 210, 90)   # 공업지역
            if "녹지" in key:
                return (210, 240, 210, 90)   # 녹지지역
            if "유통" in key:
                return (255, 240, 170, 90)   # 유통상업

            # 기본
            return (235, 235, 235, 70)

        # ---------------- 필지 loop ----------------
        for row in parcels:
            try:
                geom_bytes = row.get("geom")
                if geom_bytes is None:
                    continue
                g5186 = wkb.loads(geom_bytes)
            except Exception as e:
                print(f"[WKB ERROR] {e}")
                continue

            g3857 = shp_transform(self.tr_5186_to_3857.transform, g5186)
            if g3857.is_empty:
                continue

            if g3857.geom_type == "Polygon":
                polys = [g3857]
            elif g3857.geom_type == "MultiPolygon":
                polys = list(g3857.geoms)
            else:
                continue

            zoning_name = row.get("zoning_name")
            zoning_lclass = row.get("zoning_lclass")
            pnu = row.get("pnu")
            jibun = row.get("jibun")

            fill_color = pick_color(zoning_name, zoning_lclass)

            for poly in polys:
                coords = [proj_3857_to_px(x, y) for x, y in poly.exterior.coords]

                # 3D 느낌: 살짝 오른쪽-아래로 그림자
                shadow = [(x + 1, y + 1) for x, y in coords]
                draw.polygon(shadow, fill=(0, 0, 0, 40))

                # 본 필지 레이어
                draw.polygon(
                    coords,
                    fill=fill_color,
                    outline=(255, 255, 255, 180),
                )
                # 얇은 어두운 외곽선 한 번 더
                draw.line(coords, fill=(80, 80, 80, 160), width=1)

                # 라벨링 (지번 / 또는 PNU 뒷자리)
                centroid = poly.centroid
                cx_px, cy_px = proj_3857_to_px(centroid.x, centroid.y)
                label = jibun or (pnu[-4:] if pnu else None)

                # 너무 작은 폴리곤엔 라벨 안 찍기
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                if (max(xs) - min(xs)) < 25 or (max(ys) - min(ys)) < 18:
                    label = None

                if label:
                    draw.text((cx_px, cy_px), str(label), fill=(30, 30, 30, 220))

        # ---------------- 300m / 500m 버퍼 ----------------
        cx, cy = lonlat_to_webmerc(lon, lat)  # 중심점 3857
        center_3857 = Point(cx, cy)

        circle500 = center_3857.buffer(500, resolution=64)
        circle300 = center_3857.buffer(300, resolution=64)

        coords500 = [proj_3857_to_px(x, y) for x, y in circle500.exterior.coords]
        coords300 = [proj_3857_to_px(x, y) for x, y in circle300.exterior.coords]

        # 500m / 300m 테두리
        draw.line(coords500, fill=(255, 70, 70, 240), width=3)
        draw.line(coords300, fill=(70, 110, 255, 240), width=2)

        # 버퍼 라벨
        if coords500:
            x, y = coords500[len(coords500) // 4]
            draw.text((x, y - 15), "500m", fill=(255, 70, 70, 240))
        if coords300:
            x, y = coords300[len(coords300) // 4]
            draw.text((x, y - 15), "300m", fill=(70, 110, 255, 240))

        # 중심점 마커
        px_center_x, px_center_y = proj_3857_to_px(cx, cy)
        r = 6
        draw.ellipse(
            (px_center_x - r, px_center_y - r, px_center_x + r, px_center_y + r),
            fill=(255, 220, 70, 255),
            outline=(0, 0, 0, 255),
        )

        return base_img

    # ----------------------------
    # 5) HTML 인터랙티브 지도용 메서드
    # ----------------------------
    def generate_interactive_html(self, lon, lat, radius=500):
        """
        PNG terrain과 거의 동일한 시각을 Leaflet 인터랙티브 지도(HTML)로 생성
        """

        # -----------------------------
        # 1) parcels 가져오기
        # -----------------------------
        parcels = self.query_parcels(lon, lat, radius)

        # -----------------------------
        # 2) parcel → GeoJSON 변환
        # -----------------------------
        from shapely.geometry import mapping

        geojson_features = []

        # 5186 → 3857 → 4326 변환기
        tr_3857_to_4326 = Transformer.from_crs(3857, 4326, always_xy=True)

        for row in parcels:
            try:
                geom = wkb.loads(row["geom"])
                g3857 = shp_transform(self.tr_5186_to_3857.transform, geom)
                g4326 = shp_transform(tr_3857_to_4326.transform, g3857)

                gj = mapping(g4326)

                gj["properties"] = {
                    "pnu": row.get("pnu"),
                    "jibun": row.get("jibun"),
                    "zoning_name": row.get("zoning_name"),
                    "zoning_lclass": row.get("zoning_lclass"),
                    "zoning_mclass": row.get("zoning_mclass"),
                    "zoning_sclass": row.get("zoning_sclass"),
                }

                geojson_features.append(gj)

            except Exception as e:
                print("[GeoJSON ERROR]", e)
                continue

        geojson = {
            "type": "FeatureCollection",
            "features": geojson_features
        }

        # -----------------------------
        # 3) 300m / 500m 버퍼 GeoJSON
        # -----------------------------
        cx, cy = lonlat_to_webmerc(lon, lat)

        circle500 = Point(cx, cy).buffer(500, resolution=64)
        circle300 = Point(cx, cy).buffer(300, resolution=64)

        circle500_4326 = shp_transform(
            Transformer.from_crs(3857, 4326, always_xy=True).transform, circle500
        )
        circle300_4326 = shp_transform(
            Transformer.from_crs(3857, 4326, always_xy=True).transform, circle300
        )

        from shapely.geometry import mapping as shp_mapping
        gj_500 = shp_mapping(circle500_4326)
        gj_300 = shp_mapping(circle300_4326)

        # -----------------------------
        # 4) HTML Leaflet 페이지 구성
        # -----------------------------
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Terrain Map</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
            <style>
                html, body {{
                    height: 100%;
                    margin: 0;
                }}
                #map {{
                    height: 100%;
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>

            <script>
                var map = L.map('map').setView([{lat}, {lon}], 16);

                // DEM hillshade
                L.tileLayer.wms("https://xdworld.vworld.kr/xdworld/wms?", {{
                    layers: 'dem',
                    styles: 'hillshade',
                    format: 'image/png',
                    transparent: true,
                    attribution: 'VWorld',
                    apiKey: '{VWORLD_API_KEY}'
                }}).addTo(map);

                // 필지 GeoJSON
                var parcels = {json.dumps(geojson)};

                L.geoJSON(parcels, {{
                    style: function(feature) {{
                        var z = feature.properties.zoning_name || feature.properties.zoning_lclass || "";
                        var color = "#dddddd";

                        if (z.indexOf("상업") !== -1) color = "#ffdcbf";
                        else if (z.indexOf("주거") !== -1) color = "#dce9ff";
                        else if (z.indexOf("공업") !== -1) color = "#ffd2d2";
                        else if (z.indexOf("녹지") !== -1) color = "#d2e1d2";
                        else if (z.indexOf("유통") !== -1) color = "#fff2b3";

                        return {{
                            fillColor: color,
                            color: "#555",
                            weight: 1,
                            fillOpacity: 0.5
                        }};
                    }},
                    onEachFeature: function (feature, layer) {{
                        var p = feature.properties;
                        layer.bindPopup(
                            "<b>지번:</b> " + (p.jibun || '-') + "<br>" +
                            "<b>용도지역(대분류):</b> " + (p.zoning_lclass || '-') + "<br>" +
                            "<b>용도지역(이름):</b> " + (p.zoning_name || '-') + "<br>" +
                            "<b>PNU:</b> " + (p.pnu || '-')
                        );
                    }}
                }}).addTo(map);

                // 버퍼 500m/300m
                L.geoJSON({json.dumps(gj_500)}, {{
                    style: {{
                        color: "red",
                        weight: 2,
                        fillOpacity: 0
                    }}
                }}).addTo(map);

                L.geoJSON({json.dumps(gj_300)}, {{
                    style: {{
                        color: "blue",
                        weight: 2,
                        fillOpacity: 0
                    }}
                }}).addTo(map);

                // 중심점
                L.circleMarker([{lat}, {lon}], {{
                    radius: 6,
                    color: "yellow",
                    fillColor: "orange",
                    fillOpacity: 1
                }}).addTo(map);

            </script>
        </body>
        </html>
        """

        return html
