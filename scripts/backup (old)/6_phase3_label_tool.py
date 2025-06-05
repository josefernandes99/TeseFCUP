# scripts/6_phase3_label_tool.py
# Instead of a custom script, we might just open labelme with a directory argument:
# python scripts/6_phase3_label_tool.py data/phase2/high_res

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python 6_phase3_label_tool.py <folder_of_images>")
        sys.exit(1)
    img_dir = sys.argv[1]
    os.system(f"labelme {img_dir} --output labels/phase3/segmentation_masks")

if __name__ == "__main__":
    main()
