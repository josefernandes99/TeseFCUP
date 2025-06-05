import os

def main():
    print("=== PHASE 2: EXPORT KML & OBTAIN HIGH-RES DATA ===")
    os.system("python scripts/5_phase2_kml_export.py")

    print("\nOpen Google Earth Pro with the exported KMLs in data/phase2/kml/.")
    print("Save 8k (max res) images for each region to data/phase2/high_res/.")
    print("Then run Phase 3 for fine segmentation.\n")

if __name__ == "__main__":
    main()
