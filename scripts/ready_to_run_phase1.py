# scripts/ready_to_run_phase1.py
from a0_setup_check import setup_check
from a1_phase1_data_download import download_data
from a2_phase1_initial_labeling import initial_labeling
from a4_phase1_active_learning_loop import active_learning_loop
from a6_phase1_postprocessing import postprocessing


def main():
    print("=== Starting PythonProject Pipeline ===\n")

    setup_check()
    download_data()
    initial_labeling()
    active_learning_loop()
    postprocessing()

    print("\n=== Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    main()
