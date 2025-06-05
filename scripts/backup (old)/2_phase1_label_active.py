import os
import random
import csv
import numpy as np
import rasterio
from rasterio.transform import xy
import cv2
import logging
from rich.prompt import Prompt

LABELS_CSV = 'labels/phase1/labels.csv'
DATA_DIR = 'data/phase1/raw'
MAX_DISPLAY_SIZE = 1024  # Maximum dimension for display window
TEMP_KML_DIR = os.path.join('data', 'phase1', 'temp')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    os.makedirs(TEMP_KML_DIR, exist_ok=True)
    if not os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_file', 'x', 'y', 'lat', 'lon', 'label'])

    image_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
    if not image_files:
        logging.error("No TIF files found in data/phase1/raw.")
        return
    random.shuffle(image_files)

    logging.info("========== PHASE 1: ACTIVE LABELING (Single-Pixel Version) ==========")
    logging.info("Keyboard options: 'a' -> agriculture, 'n' -> non-ag, 'q' -> quit, ESC -> skip")

    while True:
        img_file = random.choice(image_files)

        with rasterio.open(img_file) as src:
            h, w = src.height, src.width
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            lon_center, lat_center = xy(src.transform, y, x)
            ul_lon, ul_lat = xy(src.transform, y, x, offset='ul')
            ur_lon, ur_lat = xy(src.transform, y, x, offset='ur')
            lr_lon, lr_lat = xy(src.transform, y, x, offset='lr')
            ll_lon, ll_lat = xy(src.transform, y, x, offset='ll')

            band_indices = find_rgb_band_indices(src)
            rgb = src.read(indexes=band_indices).astype(np.float32)
            rgb = np.transpose(rgb, (1, 2, 0))
            mn, mx = rgb.min(), rgb.max()
            rgb_disp = ((rgb - mn) / (mx - mn + 1e-5) * 255).clip(0, 255).astype(np.uint8)

            disp_h, disp_w = rgb_disp.shape[:2]
            scale_ratio = 1.0
            if disp_h > MAX_DISPLAY_SIZE or disp_w > MAX_DISPLAY_SIZE:
                scale_ratio = min(MAX_DISPLAY_SIZE / disp_h, MAX_DISPLAY_SIZE / disp_w)
                new_w = int(disp_w * scale_ratio)
                new_h = int(disp_h * scale_ratio)
                rgb_disp = cv2.resize(rgb_disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb_disp = np.ascontiguousarray(rgb_disp)
            x_scaled = int(x * scale_ratio)
            y_scaled = int(y * scale_ratio)
            cv2.circle(rgb_disp, (x_scaled, y_scaled), radius=5, color=(0, 0, 255), thickness=-1)

        kml_file = os.path.join(TEMP_KML_DIR, "temp_pixel.kml")
        create_pixel_kml(kml_file, ul_lon, ul_lat, ur_lon, ur_lat, lr_lon, lr_lat, ll_lon, ll_lat)

        print(f"\nFile: {img_file}")
        print(f"Selected pixel (x={x}, y={y})")
        print(f"Center coordinate: lat = {lat_center:.6f}, lon = {lon_center:.6f}")
        print("Pixel footprint (bounding box):")
        print(f"  Upper Left : lat = {ul_lat:.6f}, lon = {ul_lon:.6f}")
        print(f"  Upper Right: lat = {ur_lat:.6f}, lon = {ur_lon:.6f}")
        print(f"  Lower Right: lat = {lr_lat:.6f}, lon = {lr_lon:.6f}")
        print(f"  Lower Left : lat = {ll_lat:.6f}, lon = {ll_lon:.6f}")
        print(f"A temporary KML file has been created at: {kml_file}")

        window_name = "Labeler"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, rgb_disp)
        cv2.resizeWindow(window_name, 800, 800)

        label = None
        while True:
            # Check if window is closed unexpectedly
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logging.warning("Labeling window closed unexpectedly.")
                user_choice = Prompt.ask(
                    "Choose an option: [1] Continue labeling, [2] Save annotations and exit, [3] End labeling and proceed",
                    choices=["1", "2", "3"],
                    default="1"
                )
                if user_choice == "1":
                    logging.info("Continuing labeling session.")
                    cv2.imshow(window_name, rgb_disp)
                    cv2.resizeWindow(window_name, 800, 800)
                    continue
                elif user_choice == "2":
                    logging.info("Saving annotations and exiting labeling session.")
                    label = 'quit'
                    break
                elif user_choice == "3":
                    logging.info("Ending labeling and proceeding to next pipeline step.")
                    label = 'proceed'
                    break

            key = cv2.waitKey(50) & 0xFF
            if key == ord('a'):
                label = 'ag'
                break
            elif key == ord('n'):
                label = 'non-ag'
                break
            elif key == ord('q'):
                label = 'quit'
                break
            elif key == 27:  # ESC: skip this pixel
                label = None
                break

        cv2.destroyAllWindows()

        if label == 'quit':
            print("User chose to quit labeling.")
            break
        elif label == 'proceed':
            print("User chose to proceed to the next pipeline step.")
            break
        elif label in ['ag', 'non-ag']:
            try:
                with open(LABELS_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([img_file, x, y, f"{lat_center:.6f}", f"{lon_center:.6f}", label])
                logging.info(f"Labeled pixel as '{label}'.")
            except Exception as e:
                logging.error(f"Error writing to CSV: {e}")
        else:
            logging.info("Skipped labeling this pixel.")

    print("\nLabeling session ended.")


def find_rgb_band_indices(src):
    band_indices = {'B2': None, 'B3': None, 'B4': None}
    descs = src.descriptions
    if descs:
        for i, desc in enumerate(descs):
            if not desc:
                continue
            d_lower = desc.lower()
            if 'b2_early' in d_lower or d_lower == 'b2':
                band_indices['B2'] = i
            elif 'b3_early' in d_lower or d_lower == 'b3':
                band_indices['B3'] = i
            elif 'b4_early' in d_lower or d_lower == 'b4':
                band_indices['B4'] = i
    if None in band_indices.values():
        count = src.count
        if count < 3:
            return [1, 1, 1]
        else:
            return [1, 2, 3]
    else:
        return [band_indices['B2'] + 1, band_indices['B3'] + 1, band_indices['B4'] + 1]


def create_pixel_kml(kml_path, ul_lon, ul_lat, ur_lon, ur_lat, lr_lon, lr_lat, ll_lon, ll_lat):
    kml_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <name>Selected Pixel Footprint</name>
    <Style>
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
      <PolyStyle>
        <color>33ff0000</color>
      </PolyStyle>
    </Style>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            {ul_lon},{ul_lat},0 
            {ur_lon},{ur_lat},0 
            {lr_lon},{lr_lat},0 
            {ll_lon},{ll_lat},0 
            {ul_lon},{ul_lat},0
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
</kml>'''
    with open(kml_path, 'w') as f:
        f.write(kml_str)


if __name__ == "__main__":
    main()
