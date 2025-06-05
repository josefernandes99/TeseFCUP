# scripts/8_phase3_inference.py
import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig

MODEL_PATH  = 'models/phase3_segformer.pth'
INFER_DIR   = 'data/phase2/high_res'   # or data/phase3/inference_input
OUT_DIR     = 'data/phase3/output'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    config = SegformerConfig(num_labels=2)
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_files = [f for f in os.listdir(INFER_DIR) if f.lower().endswith(('jpg','png','jpeg','tif'))]

    for f in image_files:
        path = os.path.join(INFER_DIR, f)
        img  = Image.open(path).convert('RGB')
        original_size = img.size  # (width, height)

        # Resize to 512x512 for model input
        img_resized = img.resize((512, 512))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # CHW
        input_tensor = torch.tensor([img_array], dtype=torch.float).to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits  # shape: (batch, num_labels, H, W)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()[0]  # shape: (H, W)

        # Convert preds back to original size
        mask_resized = Image.fromarray(preds.astype(np.uint8)).resize(original_size, resample=Image.NEAREST)
        mask_array = np.array(mask_resized)

        # Create a red overlay
        overlay = np.array(img).copy()
        # Where mask == 1, overlay in red
        overlay[mask_array == 1] = [255, 0, 0]  # RGB
        overlay_img = Image.fromarray(overlay)

        # Save outputs
        base_name = os.path.splitext(f)[0]
        out_path_mask = os.path.join(OUT_DIR, f"{base_name}_mask.png")
        out_path_overlay = os.path.join(OUT_DIR, f"{base_name}_overlay.png")

        mask_resized.save(out_path_mask)
        overlay_img.save(out_path_overlay)

        print(f"Inference done for {f}, saved mask and overlay in {OUT_DIR}")

if __name__ == "__main__":
    main()
