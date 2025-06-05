import os

def main():
    print("=== PHASE 3: FINE SEGMENTATION WITH SEGFORMER ===")
    os.system("python scripts/6_phase3_label_tool.py data/phase2/high_res")
    os.system("python scripts/7_phase3_train_segformer.py")
    os.system("python scripts/8_phase3_inference.py")

    print("Phase 3 complete! Check data/phase3/output/ for segmentation results.")

if __name__ == "__main__":
    main()
