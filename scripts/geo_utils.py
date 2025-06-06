from pyproj import Transformer

WGS84 = "EPSG:4326"
TILE_CRS = "EPSG:32627"

# Setup transformers once
_to_tile = Transformer.from_crs(WGS84, TILE_CRS, always_xy=True)
_to_wgs = Transformer.from_crs(TILE_CRS, WGS84, always_xy=True)


def wgs_to_tile(lat, lon):
    """Convert WGS84 latitude/longitude to tile CRS coordinates."""
    x, y = _to_tile.transform(lon, lat)
    return x, y


def tile_to_wgs(x, y):
    """Convert tile CRS coordinates to WGS84 latitude/longitude."""
    lon, lat = _to_wgs.transform(x, y)
    return lat, lon
