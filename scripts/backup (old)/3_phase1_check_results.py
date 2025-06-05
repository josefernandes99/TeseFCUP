# scripts/3_phase1_check_results.py
import os
import csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import xy
from torchvision.models import resnet18
from rich.console import Console
from rich.progress import Progress, track

# -----------------------------
# Configuration and Paths
# -----------------------------
LABELS_CSV = "C:/Users/Vitorino/PycharmProjects/PythonProject/labels/phase1/labels.csv"
DATA_DIR = "/data/phase1/raw"
RESULTS_DIR = "C:/Users/Vitorino/PycharmProjects/PythonProject/data/phase1/results"
MODEL_DIR = "/models"
MODEL_PATH = os.path.join(MODEL_DIR, 'phase1_classifier_active.pth')
PATCH_SIZE = 64       # same patch size as used during training
STRIDE = PATCH_SIZE   # non-overlapping patches
OUTPUT_CSV = os.path.join(RESULTS_DIR, "phase1_ag_predictions.csv")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

console = Console()
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Inference Function
# -----------------------------
def inference_on_tile(image_file, model, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Slide a window over a tile and run inference on each patch.
    Returns a list of dictionaries (one per patch predicted as agricultural).
    """
    predictions = []
    with rasterio.open(image_file) as src:
        img = src.read().astype(np.float32)
        # Normalize each band independently
        for b in range(img.shape[0]):
            band = img[b]
            mn, mx = band.min(), band.max()
            img[b] = (band - mn) / (mx - mn + 1e-5)
        transform = src.transform
        height, width = src.height, src.width

    model.eval()
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(patch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            # Predict agricultural if probability for class 1 is higher
            if probs[1] > probs[0]:
                center_x = x + patch_size // 2
                center_y = y + patch_size // 2
                lon, lat = xy(transform, center_y, center_x)
                predictions.append({
                    "image_file": image_file,
                    "x": center_x,
                    "y": center_y,
                    "lon": f"{lon:.6f}",
                    "lat": f"{lat:.6f}",
                    "predicted": "ag",
                    "source": "model",
                    "uncertainty": 1 - abs(probs[0] - probs[1])
                })
    return predictions

# -----------------------------
# Overlay Image Creation
# -----------------------------
def create_overlay_image(image_file, ag_points, output_file):
    """
    Create an RGB copy of the tile (using the first three bands) with red circles at each agricultural patch center.
    """
    with rasterio.open(image_file) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
        for b in range(rgb.shape[0]):
            band = rgb[b]
            mn, mx = band.min(), band.max()
            rgb[b] = ((band - mn) / (mx - mn + 1e-5)) * 255
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.transpose(rgb, (1, 2, 0))
    # Ensure the image is contiguous in memory for OpenCV functions.
    rgb = np.ascontiguousarray(rgb)
    for point in ag_points:
        cv2.circle(rgb, (int(point["x"]), int(point["y"])), 3, (0, 0, 255), thickness=-1)
    cv2.imwrite(output_file, rgb)

# -----------------------------
# Labels Loading
# -----------------------------
def load_existing_labels():
    """
    Load labels from LABELS_CSV and return a list of entries.
    """
    entries = []
    expected = ["image_file", "x", "y", "lat", "lon", "label"]
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        data_rows = rows[1:] if rows and rows[0] == expected else rows
        for row in data_rows:
            if len(row) != 6:
                continue
            d = dict(zip(expected, row))
            entries.append(d)
    return entries, set()

def load_user_ag_labels():
    """Return user-labeled agricultural examples, marking them as source 'user'."""
    entries, _ = load_existing_labels()
    user_ag = [e for e in entries if e['label'] == 'ag']
    for e in user_ag:
        e["source"] = "user"
    return user_ag

def print_label_summary(label_entries):
    """Print a summary of user labels."""
    total = len(label_entries)
    ag = sum(1 for e in label_entries if e['label'] == 'ag')
    non_ag = total - ag
    console.print(f"\nCurrent labeled pixels: {total} (Ag: {ag}, Non-Ag: {non_ag})\n"
                  f"Required ag ratio: between 0.20 and 0.30")

# -----------------------------
# Main Routine
# -----------------------------
def main():
    console.print("[bold blue]Phase 1: Check Model Results[/bold blue]")

    # Loading trained model with a progress bar
    with Progress() as progress:
        load_task = progress.add_task("[green]Loading trained model...", total=1)
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(33, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        progress.update(load_task, advance=1)
    console.print("Model loaded.\n")

    user_ag = load_user_ag_labels()

    all_predictions = []
    image_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
    total_patches = 0
    total_ag = 0
    for image_file in track(image_files, description="Running inference on tiles..."):
        preds = inference_on_tile(image_file, model, patch_size=PATCH_SIZE, stride=STRIDE)
        with rasterio.open(image_file) as src:
            count_x = (src.width - PATCH_SIZE) // STRIDE + 1
            count_y = (src.height - PATCH_SIZE) // STRIDE + 1
        total_patches += count_x * count_y
        total_ag += len(preds)
        all_predictions.extend(preds)
        overlay_out = os.path.join(RESULTS_DIR, os.path.basename(image_file))
        create_overlay_image(image_file, preds, overlay_out)

    combined = user_ag + all_predictions

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_file", "x", "y", "lon", "lat", "predicted", "source", "uncertainty"])
        for row in combined:
            writer.writerow([row["image_file"], row["x"], row["y"], row["lon"], row["lat"],
                             row.get("predicted", "ag"), row["source"], row.get("uncertainty", "")])
    console.print(f"\nSaved combined predictions (user and model) to {OUTPUT_CSV}")

    if total_patches > 0:
        model_ag_ratio = total_ag / total_patches
    else:
        model_ag_ratio = 0
    console.print(f"Total patches processed: {total_patches}")
    console.print(f"Patches predicted as agricultural (model): {total_ag}")
    console.print(f"Model agricultural ratio (patch level): {model_ag_ratio:.2f}")

if __name__ == "__main__":
    main()
