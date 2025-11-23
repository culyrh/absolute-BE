# app/services/terrain_utils.py
import math

TILE_SIZE = 512  # VWorld도 보통 256 or 512. 직접 확인해서 맞추면 됨.

def lonlat_to_tile(lon: float, lat: float, zoom: int):
    """경위도 → WebMercator XYZ 타일 좌표"""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def tile_to_lonlat_bounds(x: int, y: int, zoom: int):
    """타일 한 장의 경위도 bbox (minlon, minlat, maxlon, maxlat)"""
    n = 2 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0

    def mercator_to_lat(t):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * t))))

    lat_top = mercator_to_lat(y / n)
    lat_bottom = mercator_to_lat((y + 1) / n)
    return lon_left, lat_bottom, lon_right, lat_top

def lonlat_to_webmerc(lon: float, lat: float):
    """경위도 → EPSG:3857 meter"""
    origin_shift = 2 * math.pi * 6378137 / 2.0
    x = lon * origin_shift / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) * 6378137
    return x, y
