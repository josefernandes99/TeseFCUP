# scripts/2_phase1_active_learning.py
import os
import random
import csv
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import rasterio
from rasterio.transform import xy, rowcol
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.progress import track, Progress

# -----------------------------
# Configuration and Paths
# -----------------------------
LABELS_CSV = os.path.join('labels', 'phase1', 'labels.csv')
DATA_DIR = os.path.join('data', 'phase1', 'raw')
TEMP_KML_DIR = os.path.join('data', 'phase1', 'temp')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'phase1_classifier_active.pth')
PATCH_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 5
ACTIVE_BATCH_SIZE = 50    # Number of candidates to label per round
MAX_ROUNDS = 5            # Maximum active learning rounds
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# (NDVI/NDWI settings are legacy here and not used for candidate filtering)
NDVI_THRESHOLD = 0.3
NDWI_THRESHOLD = 0.1

# For balancing, our target is 25% agricultural examples.
DESIRED_AG_RATIO = 0.25
# We require the training set to be balanced between 20% and 30% agricultural examples.
ACCEPTABLE_AG_RATIO_LOWER = 0.20
ACCEPTABLE_AG_RATIO_UPPER = 0.30
MIN_AG_REQUIRED = 100     # Minimum number of ag examples required before training

console = Console()

# Ensure necessary directories exist
os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
os.makedirs(TEMP_KML_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# (Legacy) NDVI Dynamic Threshold Function
# -----------------------------
def get_dynamic_ndvi_bounds(image_file):
    """
    (Legacy function; not used for candidate sampling in this version.)
    """
    dt = None
    with rasterio.open(image_file) as src:
        tags = src.tags()
        ts = tags.get('system:time_start') or tags.get('DATE') or tags.get('TIMESTAMP')
        if ts is not None:
            try:
                ts_int = int(ts)
                dt = datetime.datetime.fromtimestamp(ts_int / 1000, datetime.timezone.utc)
            except Exception:
                try:
                    dt = datetime.datetime.fromisoformat(ts)
                except Exception:
                    dt = None
    if dt is not None:
        month = dt.month
        if 3 <= month <= 6:
            lower_bound = 0.2
        elif 8 <= month <= 10:
            lower_bound = 0.3
        else:
            lower_bound = 0.25
    else:
        lower_bound = 0.25
    upper_bound = 0.85
    return lower_bound, upper_bound


# -----------------------------
# Helper Functions
# -----------------------------
def find_rgb_band_indices(src):
    """Return a list of 1-indexed band indices for RGB (fallback to first three)."""
    band_indices = {'B2': None, 'B3': None, 'B4': None}
    descs = src.descriptions
    if descs:
        for i, desc in enumerate(descs):
            if not desc:
                continue
            d_lower = desc.lower()
            if 'b2' in d_lower:
                band_indices['B2'] = i
            elif 'b3' in d_lower:
                band_indices['B3'] = i
            elif 'b4' in d_lower:
                band_indices['B4'] = i
    if None in band_indices.values():
        return [1, 2, 3]
    else:
        return [band_indices['B2'] + 1, band_indices['B3'] + 1, band_indices['B4'] + 1]


def load_patch_center(image_file, x, y, patch_size=PATCH_SIZE):
    """Extract and normalize a patch with the given (x,y) pixel at its center."""
    with rasterio.open(image_file) as src:
        img = src.read()  # shape: (channels, height, width)
        C, H, W = img.shape
        half = patch_size // 2
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        if x1 + patch_size > W:
            x1 = W - patch_size
        if y1 + patch_size > H:
            y1 = H - patch_size
        patch = img[:, y1:y1+patch_size, x1:x1+patch_size]
        patch = patch.astype(np.float32)
        mn, mx = patch.min(), patch.max()
        patch = (patch - mn) / (mx - mn + 1e-5)
        return torch.from_numpy(patch)


def load_existing_labels():
    """
    Load labels from CSV. Returns a list of entries and a set of (image_file, x, y) keys.
    """
    entries = []
    labeled_set = set()
    expected = ["image_file", "x", "y", "lat", "lon", "label"]
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            with open(LABELS_CSV, 'w', newline='') as f:
                csv.writer(f).writerow(expected)
            data_rows = []
        else:
            data_rows = rows[1:] if rows[0] == expected else rows
            for row in data_rows:
                if len(row) != 6:
                    console.print(f"[bold red]Skipping row due to incorrect number of columns: {row}[/bold red]")
                    continue
                d = dict(zip(expected, row))
                try:
                    labeled_set.add((d['image_file'], int(d['x']), int(d['y'])))
                except Exception as e:
                    console.print(f"[bold red]Error processing row: {row}. Error: {e}[/bold red]")
                    continue
                entries.append(d)
    else:
        with open(LABELS_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(expected)
    return entries, labeled_set


def current_ag_ratio(label_entries):
    """Return the ratio of agricultural examples."""
    if not label_entries:
        return 0
    count_ag = sum(1 for e in label_entries if e['label'] == 'ag')
    return count_ag / len(label_entries)


def prepare_balanced_dataset(label_entries, desired_ratio=DESIRED_AG_RATIO):
    """
    Use all ag examples and sample non-ag examples to achieve the desired ratio.
    """
    ag_entries = [e for e in label_entries if e['label'] == 'ag']
    non_ag_entries = [e for e in label_entries if e['label'] == 'non-ag']
    if len(ag_entries) < MIN_AG_REQUIRED:
        console.print(f"[bold red]Insufficient ag examples ({len(ag_entries)} found, need {MIN_AG_REQUIRED}).[/bold red]")
        return None
    target_non_ag = int((1 - desired_ratio) / desired_ratio * len(ag_entries))
    sampled_non_ag = random.sample(non_ag_entries, target_non_ag) if len(non_ag_entries) > target_non_ag else non_ag_entries
    balanced = ag_entries + sampled_non_ag
    random.shuffle(balanced)
    return balanced


class CenterPatchDataset(Dataset):
    """Dataset to extract patches from images based on center pixel coordinates."""
    def __init__(self, entries, patch_size=PATCH_SIZE):
        self.entries = entries
        self.patch_size = patch_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        image_file = e['image_file']
        x = int(e['x'])
        y = int(e['y'])
        label = 1 if e['label'] == 'ag' else 0
        patch = load_patch_center(image_file, x, y, self.patch_size)
        return patch, label


def train_model_round(label_entries):
    """Train a model on a balanced subset of labeled data and return the trained model."""
    balanced_entries = prepare_balanced_dataset(label_entries)
    if balanced_entries is None:
        return None
    console.print("[bold green]Training model on balanced labeled data...[/bold green]")
    dataset = CenterPatchDataset(balanced_entries, PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(33, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        with Progress() as progress:
            task = progress.add_task(f"Epoch {epoch+1}/{EPOCHS}", total=len(dataloader))
            for x, y in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress.advance(task)
        console.print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")
    return model


def create_pixel_kml(kml_path, image_file, x, y):
    """Generate a KML file for the candidate pixel's ground footprint."""
    with rasterio.open(image_file) as src:
        ul_lon, ul_lat = xy(src.transform, y, x, offset='ul')
        ur_lon, ur_lat = xy(src.transform, y, x, offset='ur')
        lr_lon, lr_lat = xy(src.transform, y, x, offset='lr')
        ll_lon, ll_lat = xy(src.transform, y, x, offset='ll')
    kml_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <name>Candidate Pixel Footprint</name>
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


def sample_candidates(model, labeled_set, num_candidates_per_image=5, use_model=True):
    """
    Randomly sample candidate pixels from each raw image.
    If a model is provided, compute an uncertainty score.
    """
    candidates = []
    image_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.tif')]
    for image_file in track(image_files, description="Sampling candidate pixels from images..."):
        with rasterio.open(image_file) as src:
            H, W = src.height, src.width
        for _ in range(num_candidates_per_image):
            x = random.randint(0, W - 1)
            y = random.randint(0, H - 1)
            if (image_file, x, y) in labeled_set:
                continue
            patch = load_patch_center(image_file, x, y, PATCH_SIZE)
            patch = patch.unsqueeze(0).to(DEVICE)
            if use_model and model is not None:
                model.eval()
                with torch.no_grad():
                    outputs = model(patch)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                uncertainty = 1 - abs(probs[0] - probs[1])
            else:
                uncertainty = 1.0
            candidates.append({
                'image_file': image_file,
                'x': x,
                'y': y,
                'uncertainty': uncertainty,
                'patch': patch.cpu().squeeze(0).numpy()
            })
    return candidates


def label_candidate(candidate, label_entries):
    """
    Display the full tile with the candidate pixel highlighted and generate a fixed KML file.
    Wait indefinitely until you provide a label.
    """
    kml_filename = "temp_kml_file.kml"  # fixed filename to overwrite previous one
    kml_path = os.path.join(TEMP_KML_DIR, kml_filename)
    create_pixel_kml(kml_path, candidate['image_file'], candidate['x'], candidate['y'])

    with rasterio.open(candidate['image_file']) as src:
        band_indices = find_rgb_band_indices(src)
        rgb = src.read(indexes=band_indices).astype(np.float32)
        rgb = np.transpose(rgb, (1, 2, 0))
        mn, mx = rgb.min(), rgb.max()
        rgb_disp = ((rgb - mn) / (mx - mn + 1e-5) * 255).clip(0, 255).astype(np.uint8)
        max_display_size = 1024
        disp_h, disp_w = rgb_disp.shape[:2]
        scale_ratio = 1.0
        if disp_h > max_display_size or disp_w > max_display_size:
            scale_ratio = min(max_display_size/disp_h, max_display_size/disp_w)
            new_w = int(disp_w * scale_ratio)
            new_h = int(disp_h * scale_ratio)
            rgb_disp = cv2.resize(rgb_disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb_disp = np.ascontiguousarray(rgb_disp)
    cv2.circle(rgb_disp, (int(candidate['x'] * scale_ratio), int(candidate['y'] * scale_ratio)), 5, (0,0,255), thickness=-1)
    cv2.putText(rgb_disp, "a: ag   n: non-ag   q/ESC: skip", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(rgb_disp, f"KML: {kml_filename}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    window_name = "Active Learning Label"
    cv2.imshow(window_name, rgb_disp)
    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
    if key == ord('a'):
        label = 'ag'
    elif key == ord('n'):
        label = 'non-ag'
    else:
        label = None
    cv2.destroyWindow(window_name)
    if label == 'non-ag' and current_ag_ratio(label_entries) < DESIRED_AG_RATIO:
        console.print("[bold yellow]Skipping non-ag label because more ag examples are needed.[/bold yellow]")
        return None
    return label


def manual_ag_input():
    """
    Prompts for geographic coordinates (longitude and latitude) and automatically finds the tile that contains them.
    Converts the coordinate into pixel indices using the tile's georeference and returns a new label entry.
    """
    console.print("[bold yellow]Please enter the geographic coordinates for an agricultural crop (longitude, latitude):[/bold yellow]")
    try:
        lon = float(input("Longitude: ").strip())
        lat = float(input("Latitude: ").strip())
    except Exception:
        console.print("[bold red]Invalid coordinate input.[/bold red]")
        return None
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.tif'):
            continue
        image_file = os.path.join(DATA_DIR, filename)
        with rasterio.open(image_file) as src:
            bounds = src.bounds  # left, bottom, right, top
            if bounds.left <= lon <= bounds.right and bounds.bottom <= lat <= bounds.top:
                row_idx, col_idx = rowcol(src.transform, lon, lat)
                confirm_lon, confirm_lat = xy(src.transform, row_idx, col_idx)
                console.print(f"Found tile: {filename}")
                console.print(f"Pixel coordinates: x (col) = {col_idx}, y (row) = {row_idx}")
                console.print(f"Tile ground coordinates: lon = {confirm_lon:.6f}, lat = {confirm_lat:.6f}")
                return {
                    'image_file': image_file,
                    'x': col_idx,
                    'y': row_idx,
                    'lat': f"{confirm_lat:.6f}",
                    'lon': f"{confirm_lon:.6f}",
                    'label': 'ag'
                }
    console.print("[bold red]No tile found that contains the provided coordinates.[/bold red]")
    return None


def print_label_summary(label_entries):
    """Print the current summary of labeled pixels and ratios."""
    total = len(label_entries)
    if total == 0:
        ag_ratio = 0
    else:
        ag_ratio = current_ag_ratio(label_entries)
    non_ag_ratio = 1 - ag_ratio
    console.print(f"\nCurrent labeled pixels: {total} (Ag: {ag_ratio:.2f}, Non-Ag: {non_ag_ratio:.2f})")
    console.print(f"Required: Ag ratio between {ACCEPTABLE_AG_RATIO_LOWER:.2f} and {ACCEPTABLE_AG_RATIO_UPPER:.2f}\n")


# -----------------------------
# Main Interactive Loop
# -----------------------------
def main():
    label_entries, labeled_set = load_existing_labels()

    while True:
        print_label_summary(label_entries)
        console.print("Main Menu:")
        console.print("  [1] Continue labeling")
        console.print("  [2] Proceed to training the model")
        main_choice = input("Enter choice (1 or 2): ").strip()
        if main_choice == "2":
            current_ratio = current_ag_ratio(label_entries)
            if current_ratio < ACCEPTABLE_AG_RATIO_LOWER or current_ratio > ACCEPTABLE_AG_RATIO_UPPER:
                console.print(
                    f"[bold red]Cannot proceed to training: agricultural ratio must be between {ACCEPTABLE_AG_RATIO_LOWER:.2f} and {ACCEPTABLE_AG_RATIO_UPPER:.2f}.[/bold red]")
                continue
            else:
                break
        elif main_choice == "1":
            console.print("Labeling Menu:")
            console.print("  [1] Add new agricultural crop data (manual input)")
            console.print("  [2] Add global data (random sampling)")
            console.print("  [3] Return to main menu")
            sub_choice = input("Enter choice (1, 2, or 3): ").strip()
            if sub_choice == "1":
                manual_entry = manual_ag_input()
                if manual_entry:
                    key = (manual_entry['image_file'], int(manual_entry['x']), int(manual_entry['y']))
                    if key in labeled_set:
                        console.print("[bold yellow]This pixel is already labeled. Skipping.[/bold yellow]")
                    else:
                        label_entries.append(manual_entry)
                        labeled_set.add(key)
                        with open(LABELS_CSV, 'a', newline='') as f:
                            csv.writer(f).writerow([manual_entry['image_file'], manual_entry['x'],
                                                      manual_entry['y'], manual_entry['lat'],
                                                      manual_entry['lon'], manual_entry['label']])
                        console.print("[bold green]Agricultural label added.[/bold green]")
                else:
                    console.print("[bold red]Manual input failed. Please try again.[/bold red]")
            elif sub_choice == "2":
                candidates = sample_candidates(None, labeled_set, num_candidates_per_image=5, use_model=False)
                if not candidates:
                    console.print("[bold red]No candidate pixels available. Check your raw imagery.[/bold red]")
                else:
                    for candidate in candidates:
                        console.print(f"Candidate from [cyan]{candidate['image_file']}[/cyan] at (x={candidate['x']}, y={candidate['y']})")
                        label = label_candidate(candidate, label_entries)
                        if label is not None:
                            with rasterio.open(candidate['image_file']) as src:
                                lon, lat = xy(src.transform, candidate['y'], candidate['x'])
                            new_entry = {
                                'image_file': candidate['image_file'],
                                'x': str(candidate['x']),
                                'y': str(candidate['y']),
                                'lat': f"{lat:.6f}",
                                'lon': f"{lon:.6f}",
                                'label': label
                            }
                            key = (candidate['image_file'], candidate['x'], candidate['y'])
                            if key not in labeled_set:
                                label_entries.append(new_entry)
                                labeled_set.add(key)
                                with open(LABELS_CSV, 'a', newline='') as f:
                                    csv.writer(f).writerow([new_entry['image_file'], new_entry['x'],
                                                              new_entry['y'], new_entry['lat'],
                                                              new_entry['lon'], new_entry['label']])
                                console.print("[bold green]Label added.[/bold green]")
                    console.print("[bold green]Finished global candidate labeling. Returning to main menu.[/bold green]")
            elif sub_choice == "3":
                continue  # Return to main menu
            else:
                console.print("[bold red]Invalid choice in labeling menu.[/bold red]")
        else:
            console.print("[bold red]Invalid main menu choice. Please enter 1 or 2.[/bold red]")

    # Proceed to training the model
    console.print("[bold blue]Proceeding to training...[/bold blue]")
    model = train_model_round(label_entries)
    if model is None:
        console.print("[bold red]Training aborted. Please add more agricultural examples manually.[/bold red]")
    else:
        torch.save(model.state_dict(), MODEL_PATH)
        console.print(f"Model saved to [bold]{MODEL_PATH}[/bold].")
    console.print("[bold blue]Phase 1 Complete! Proceed to Phase 2 (KML export & high-res imagery) next.[/bold blue]")


if __name__ == "__main__":
    main()
