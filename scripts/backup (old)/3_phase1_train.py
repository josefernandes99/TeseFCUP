import os
import csv
import torch
import random
import numpy as np
import rasterio
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LABELS_CSV = 'labels/phase1/labels.csv'
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def augment_patch(patch):
    # Data augmentation: random horizontal flip, vertical flip, and rotation (0, 90, 180, or 270 degrees)
    if random.random() > 0.5:
        patch = np.flip(patch, axis=2)  # horizontal flip
    if random.random() > 0.5:
        patch = np.flip(patch, axis=1)  # vertical flip
    k = random.randint(0, 3)
    patch = np.rot90(patch, k, axes=(1, 2))
    return patch

def load_patch_for_training(image_file, label, patch_size=64):
    with rasterio.open(image_file) as src:
        img = src.read()  # shape: (33, height, width)
        c, h, w = img.shape
        if w < patch_size or h < patch_size:
            raise ValueError(f"Image {image_file} is too small for the patch size {patch_size}.")
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        patch = img[:, y:y+patch_size, x:x+patch_size]
    patch = patch.astype(np.float32)
    # Apply data augmentation
    patch = augment_patch(patch)
    mn, mx = patch.min(), patch.max()
    patch = (patch - mn) / (mx - mn + 1e-5)
    y_label = 1 if label == 'ag' else 0
    return patch, y_label

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = np.stack(xs, axis=0)
    xs = torch.from_numpy(xs)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

def main():
    data_entries = []
    try:
        with open(LABELS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'image_file' not in row:
                    for key in row.keys():
                        if key.lower().strip() in ['file', 'filename', 'image']:
                            row['image_file'] = row[key]
                            break
                if 'image_file' in row and row.get('label'):
                    data_entries.append(row)
                else:
                    logging.warning("Skipping row due to missing image_file or label: %s", row)
    except Exception as e:
        logging.error("Error reading CSV: %s", e)
        return

    if not data_entries:
        logging.error("No valid labeled entries found in CSV. Please run the labeling phase first.")
        return

    random.shuffle(data_entries)
    split_idx = int(0.8 * len(data_entries))
    train_entries = data_entries[:split_idx]
    val_entries   = data_entries[split_idx:]

    class PatchDataset(torch.utils.data.Dataset):
        def __init__(self, entries):
            self.entries = entries
        def __len__(self):
            return len(self.entries)
        def __getitem__(self, idx):
            e = self.entries[idx]
            patch, y_label = load_patch_for_training(e['image_file'], e['label'])
            return patch, y_label

    train_ds = PatchDataset(train_entries)
    val_ds   = PatchDataset(val_entries)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                               shuffle=True, collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
                                               shuffle=False, collate_fn=collate_fn)

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(33, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        with Progress("[progress.description]{task.description}", BarColumn(), TimeRemainingColumn()) as progress:
            task_train = progress.add_task(f"Epoch {epoch+1}/{EPOCHS} - Training", total=num_batches)
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress.advance(task_train)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            num_val_batches = len(val_loader)
            with Progress("[progress.description]{task.description}", BarColumn(), TimeRemainingColumn()) as progress:
                task_val = progress.add_task(f"Epoch {epoch+1}/{EPOCHS} - Validation", total=num_val_batches)
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, dim=1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                    progress.advance(task_val)
        logging.info(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss/len(train_loader):.4f}, "
                     f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.0*correct/total:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/phase1_classifier.pth')
    logging.info("Phase1 classifier (multi-temporal) saved.")

if __name__ == "__main__":
    main()
