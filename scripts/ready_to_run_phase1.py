# scripts/ready_to_run_phase1.py
import glob
import json
import os

from a0_setup_check import setup_check
from a1_phase1_data_download import download_data
from a2_phase1_initial_labeling import initial_labeling
from a4_phase1_active_learning_loop import active_learning_loop
from a6_phase1_postprocessing import postprocessing
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
                # Checkpoint inside loop via callback
                def cb(r, nr, mc):
                    save_checkpoint("active_learning_loop", round=r, total_rounds=nr, model_choice=mc)

                active_learning_loop(al_start_round, al_total_rounds, al_model_choice, checkpoint_cb=cb)
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