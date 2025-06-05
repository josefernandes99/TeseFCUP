# CropSegProject

CropSegProject is an end-to-end pipeline for detecting and segmenting agricultural fields using satellite imagery. The project is divided into three phases:

1. **Phase 1: Coarse Detection** – Uses medium-resolution, multi-temporal imagery (from Google Earth Engine) to detect broad areas of agricultural activity.
2. **Phase 2: High-Resolution Imagery Extraction** – Converts the coarse detection masks into geographic polygons (via KML export) and guides you to manually download high-resolution (8K) images using Google Earth Pro.
3. **Phase 3: Fine Segmentation** – Uses the high-resolution imagery along with pixel-level annotations (created with a tool like LabelMe) to train a SegFormer-based model for detailed crop segmentation.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Phase 1: Coarse Detection](#phase-1-coarse-detection)
   - [Phase 2: High-Resolution Imagery Extraction](#phase-2-high-resolution-imagery-extraction)
   - [Phase 3: Fine Segmentation](#phase-3-fine-segmentation)
4. [Active Learning & Iterations](#active-learning--iterations)
5. [Future Considerations](#future-considerations)
6. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
CropSegProject/
├── data/
│   ├── phase1/
│   │   ├── raw/                # Downloaded medium-res imagery (GEE)
│   │   ├── temp/               # Temporary files used during processing
│   │   └── processed/          # Inference masks and processed outputs from Phase 1
│   ├── phase2/
│   │   ├── kml/                # KML files exported from Phase 1 (agriculture polygons)
│   │   └── high_res/           # High-resolution (8K) images saved from Google Earth Pro
│   └── phase3/
│       └── output/             # Segmentation outputs (masks and overlays)
├── labels/
│   ├── phase1/
│   │   └── labels.csv          # CSV file with single-pixel labels ("ag"/"non-ag")
│   └── phase3/
│       └── segmentation_masks/ # Pixel-level masks generated using LabelMe for segmentation
├── scripts/
│   ├── 0_setup_check.py
│   ├── 1_phase1_data_download.py
│   ├── 2_phase1_label_active.py
│   ├── 3_phase1_train.py
│   ├── 4_phase1_inference.py
│   ├── 5_phase2_kml_export.py
│   ├── 6_phase3_label_tool.py
│   ├── 7_phase3_train_segformer.py
│   └── 8_phase3_inference.py
├── ready_to_run_phase1.py
├── ready_to_run_phase2.py
├── ready_to_run_phase3.py
├── README.md
└── requirements.txt
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUser/CropSegProject.git
   cd CropSegProject
   ```

2. **Create a Conda environment (recommended):**
   ```bash
   conda create -n crops python=3.9 -y
   conda activate crops
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Authenticate with Google Earth Engine:**
   ```bash
   earthengine authenticate
   ```
   Follow the on-screen instructions to grant access.

---

## Usage

### Phase 1: Coarse Detection

Run the ready-to-run script:
```bash
python ready_to_run_phase1.py
```
This phase performs:
1. **Setup Check:** Verifies that Earth Engine is properly initialized.
2. **Data Download:** Fetches multi-temporal Sentinel-2 imagery for your region (Cape Verde by default) and splits the area into tiles.
3. **Active Labeling:** Displays random image patches with a marked pixel for you to label as "ag" (agriculture) or "non-ag."
4. **Training:** Trains a ResNet18-based classifier (modified for 33 channels) using the labeled patches.
5. **Inference:** Applies the classifier to the full set of tiles to generate a binary mask indicating agricultural areas.

---

### Phase 2: High-Resolution Imagery Extraction

Run the ready-to-run script:
```bash
python ready_to_run_phase2.py
```
This phase performs:
1. **KML Export:** Converts the binary ag masks from Phase 1 into geographic polygons (saved as KML files in `data/phase2/kml/`).
2. **Manual Step:** Open these KML files in Google Earth Pro. Use the polygons to identify and save high-resolution (8K) images into `data/phase2/high_res/`.

---

### Phase 3: Fine Segmentation

Run the ready-to-run script:
```bash
python ready_to_run_phase3.py
```
This phase performs:
1. **Segmentation Labeling:** Launches a labeling tool (LabelMe) to annotate the high-res images. The output masks are saved in `labels/phase3/segmentation_masks/`.
2. **Training SegFormer:** Trains a SegFormer-based segmentation model using the high-res images and your pixel-level annotations.
3. **Inference:** Applies the trained model to the high-res images to produce detailed crop segmentation masks and red-overlay images, which are saved in `data/phase3/output/`.

---

## Troubleshooting

- **Earth Engine Authentication:** Ensure you’ve successfully run `earthengine authenticate` and that your credentials are correctly set up.
- **Memory Issues:** If you encounter out-of-memory errors during training or inference, try reducing batch sizes or using smaller image resolutions.
- **Coordinate Alignment:** If KML polygons appear misaligned in Google Earth Pro, check the CRS settings and reprojection steps.
- **Slow Processing:** For very large images, consider using a sliding window approach to speed up inference.

---