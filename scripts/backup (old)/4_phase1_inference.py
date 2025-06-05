import os
import numpy as np
import torch
import rasterio
import torch.nn as nn
from torchvision.models import resnet18
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'models/phase1_classifier.pth'
DATA_DIR   = 'data/phase1/raw'
OUT_DIR    = 'data/phase1/processed/inference_masks'
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file {MODEL_PATH} not found. Please run the training phase (3_phase1_train.py) first.")
        return

    model = resnet18()
    model.conv1 = nn.Conv2d(33, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    PATCH_SIZE = 256
    BATCH_SIZE = 16

    tif_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
    for tif in tif_files:
        in_path = os.path.join(DATA_DIR, tif)
        logging.info(f"Running inference on {in_path}")
        with rasterio.open(in_path) as src:
            profile = src.profile
            img_data = src.read()  # shape: (33, H, W)
            C, H, W = img_data.shape

        ag_mask = np.zeros((H, W), dtype=np.uint8)
        x_positions = list(range(0, W, PATCH_SIZE))
        y_positions = list(range(0, H, PATCH_SIZE))
        patch_coords = []
        patch_tensors = []

        def flush_batch():
            nonlocal patch_coords, patch_tensors, ag_mask
            if not patch_tensors:
                return
            batch_tensor = torch.stack(patch_tensors, dim=0).to(DEVICE)
            with torch.no_grad():
                outputs = model(batch_tensor)
                preds = torch.argmax(outputs, dim=1)
            for (y0, x0), pred_class in zip(patch_coords, preds.cpu().numpy()):
                val = 255 if pred_class == 1 else 0
                ag_mask[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE] = val
            patch_coords.clear()
            patch_tensors.clear()

        total_patches = sum(1 for y0 in y_positions for x0 in x_positions if (y0 + PATCH_SIZE <= H and x0 + PATCH_SIZE <= W))
        with Progress("[progress.description]{task.description}", BarColumn(), TimeRemainingColumn()) as progress:
            task_inf = progress.add_task(f"Processing {tif}", total=total_patches)
            for y0 in y_positions:
                for x0 in x_positions:
                    patch = img_data[:, y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
                    if patch.shape[1] < PATCH_SIZE or patch.shape[2] < PATCH_SIZE:
                        progress.advance(task_inf)
                        continue
                    patch = patch.astype(np.float32)
                    mn, mx = patch.min(), patch.max()
                    patch_norm = (patch - mn) / (mx - mn + 1e-5)
                    patch_tensors.append(torch.from_numpy(patch_norm))
                    patch_coords.append((y0, x0))
                    if len(patch_tensors) == BATCH_SIZE:
                        flush_batch()
                    progress.advance(task_inf)
            flush_batch()

        out_path = os.path.join(OUT_DIR, tif.replace('.tif', '_ag_mask.tif'))
        out_profile = profile.copy()
        out_profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(ag_mask, 1)
        logging.info(f"Inference done for {tif}, saved to: {out_path}")

if __name__ == "__main__":
    main()
