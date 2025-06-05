# scripts/7_phase3_train_segformer.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn as nn
import torch.optim as optim

IMAGE_DIR = 'data/phase2/high_res'
MASK_DIR  = 'labels/phase3/segmentation_masks'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE= 2
EPOCHS    = 10

class CropSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Assume each image has a corresponding mask with the same filename base
        self.images = []
        for file in os.listdir(img_dir):
            if file.lower().endswith(('png','jpg','jpeg','tif')):
                base_name = os.path.splitext(file)[0]
                # labelme typically saves masks as the same base name + .png
                mask_path = os.path.join(mask_dir, f"{base_name}.png")
                if os.path.exists(mask_path):
                    self.images.append(base_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        base_name = self.images[idx]
        # Attempt to read as .jpg first; adapt if your images are .tif, etc.
        possible_extensions = ['.jpg', '.png', '.jpeg', '.tif']
        img_path = None
        for ext in possible_extensions:
            candidate = os.path.join(self.img_dir, f"{base_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            raise FileNotFoundError(f"No image file found for base {base_name}")

        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # single-channel

        image = image.resize((512, 512))
        mask  = mask.resize((512, 512))

        image = np.array(image).astype(np.float32) / 255.0
        mask  = np.array(mask).astype(np.int64)  # 0=background, 1=ag

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

def main():
    dataset = CropSegDataset(IMAGE_DIR, MASK_DIR)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Configure a Segformer for 2 classes (background + crop)
    config = SegformerConfig(
        num_labels=2,
        # Optionally pick a pretrained model name:
        # 'nvidia/segformer-b0-finetuned-ade-512-512', etc.
    )
    model = SegformerForSemanticSegmentation(config)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks  = masks.to(DEVICE)
                outputs = model(pixel_values=images, labels=masks)
                loss = outputs.loss
                val_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

    # Save final model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/phase3_segformer.pth')
    print("Phase3 SegFormer model saved.")

if __name__ == "__main__":
    main()
