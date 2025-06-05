import os
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, Polygon
from rasterio.features import shapes

IN_MASK_DIR = 'data/phase1/processed/inference_masks'
OUT_KML_DIR = 'data/phase2/kml'
os.makedirs(OUT_KML_DIR, exist_ok=True)

def raster_to_polygons(raster_path):
    with rasterio.open(raster_path) as src:
        mask_data = src.read(1)
        transform = src.transform
        polygons = []
        for shp, val in shapes(mask_data, transform=transform):
            if val == 255:  # farmland
                polygons.append(shape(shp))
        return polygons, src.crs

def main():
    mask_files = [f for f in os.listdir(IN_MASK_DIR) if f.endswith('_ag_mask.tif')]
    for mf in mask_files:
        in_path = os.path.join(IN_MASK_DIR, mf)
        polygons, crs = raster_to_polygons(in_path)
        if len(polygons) == 0:
            print(f"No farmland polygons found in {mf}")
            continue

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        out_kml = os.path.join(OUT_KML_DIR, mf.replace('_ag_mask.tif', '.kml'))
        gdf.to_file(out_kml, driver='KML')
        print(f"Saved KML polygons: {out_kml}")

if __name__ == "__main__":
    main()
