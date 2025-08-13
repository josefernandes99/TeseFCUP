# scripts/ready_to_run_phase1.py
import glob
import json
import os

from a0_setup_check import setup_check
from a1_phase1_data_download import download_data
from a2_phase1_initial_labeling import (
    initial_labeling,
    ensure_labels_file,
    ensure_evaluate_file,
)
from a4_phase1_active_learning_loop import active_learning_loop, collect_user_hyperparams
from a6_phase1_postprocessing import postprocessing
from grid_search import run_grid_search
from config import RAW_DATA_DIR, CHECKPOINT_FILE

STEP_ORDER = [
    "setup_check",
    "download_data",
    "initial_labeling",
    "active_learning_loop",
    "postprocessing",
]

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
            step = data.get("step")
            if step in STEP_ORDER:
                return data
        except Exception:
            pass
    return None

def save_checkpoint(step, **extra):
    data = {"step": step}
    data.update(extra)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

def main():
    print("=== Starting PythonProject Pipeline ===\n")
    print("Note: computing 5x5 GLCM textures and global normalization can be slow; this is the main start-up cost.\n")
    ensure_labels_file()
    ensure_evaluate_file()
    mw = None
    try:
        cp = load_checkpoint()
        if cp:
            step = cp.get("step")
            msg = f"Resume from checkpoint at '{step}'"
            if step == "active_learning_loop" and cp.get("round"):
                msg += f" (round {cp['round']})"
            msg += "? (y/n) => "
            ans = input(msg).strip().lower()
            if not ans.startswith("y"):
                clear_checkpoint()
                cp = None

        start_step = cp["step"] if cp else STEP_ORDER[0]

        step_idx = STEP_ORDER.index(start_step)

        al_start_round = cp.get("round", 1) if cp and start_step == "active_learning_loop" else 1
        al_total_rounds = cp.get("total_rounds") if cp else None
        al_model_choice = cp.get("model_choice") if cp else None
        al_params = cp.get("params") if cp else None

        for step in STEP_ORDER[step_idx:]:
            if step == "setup_check":
                save_checkpoint(step)
                setup_check()
            elif step == "download_data":
                save_checkpoint(step)
                if not glob.glob(os.path.join(RAW_DATA_DIR, "*.tif")):
                    download_data()
                else:
                    print("Raw data already present; skipping download.")
            elif step == "initial_labeling":
                save_checkpoint(step)
                initial_labeling()
            elif step == "active_learning_loop":
                save_checkpoint(step)
                mode = input("Hyper-parameter mode: [1] grid search, [2] specify => ").strip()
                if mode == "1":
                    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest")
                    models = ["ResNet", "SVM", "RandomForest"]
                    ch = input("=> ").strip()
                    mchoice = models[int(ch) - 1] if ch in ["1", "2", "3"] else "RandomForest"
                    run_grid_search(mchoice)
                    clear_checkpoint()
                    return
                else:
                    def cb(r, nr, mc, params):
                        save_checkpoint(
                            "active_learning_loop",
                            round=r,
                            total_rounds=nr,
                            model_choice=mc,
                            params=params,
                        )
                    mchoice = al_model_choice
                    if mchoice is None:
                        print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest")
                        models = ["ResNet", "SVM", "RandomForest"]
                        ch = input("=> ").strip()
                        if ch in ["1", "2", "3"]:
                            mchoice = models[int(ch) - 1]
                        else:
                            mchoice = "RandomForest"
                    params = al_params or collect_user_hyperparams(mchoice)
                    active_learning_loop(
                        al_start_round,
                        al_total_rounds,
                        mchoice,
                        checkpoint_cb=cb,
                        use_grid_search=False,
                        user_params=params,
                    )
            elif step == "postprocessing":
                save_checkpoint(step)
                postprocessing()

        clear_checkpoint()
        print("\n=== Pipeline Completed Successfully! ===")
    finally:
        if mw is not None:
            try:
                mw.stop()
            except Exception:
                pass

if __name__ == "__main__":
    main()
