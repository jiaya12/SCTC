"""
Run the full pipeline end to end: both tasks, the figures, and the tests.

    python run_all.py            # uses SCTC_DATA_DIR (default ./data)
    python run_all.py --demo     # generates synthetic data and runs on it
"""
import os
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)


def main():
    py = sys.executable
    if "--demo" in sys.argv:
        run([py, "make_demo_data.py"])
        os.environ["SCTC_DATA_DIR"] = "./demo_data"
    run([py, "run_vaso.py"])
    run([py, "run_septic.py"])
    run([py, "reproduce_figures.py"])
    run([py, "tests/test_sctc.py"])


if __name__ == "__main__":
    main()
