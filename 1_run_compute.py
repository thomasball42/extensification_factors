import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    Path("beta_scripts") / "_compute_beta_crops.py",
    Path("beta_scripts") / "_compute_beta_crops_linear.py",
    Path("beta_scripts") / "_compute_beta_animals.py",
    Path("beta_scripts") / "_compute_beta_animals_linear.py",
]

if __name__ == "__main__":
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(SCRIPTS)}] Running {script}\n{'=' * 70}")
        start = time.time()
        result = subprocess.run([sys.executable, script], cwd=BASE_DIR)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n{script} failed (exit code {result.returncode}) after {elapsed:.1f}s -- stopping.")
            sys.exit(result.returncode)
        print(f"\n{script} finished in {elapsed:.1f}s")

    print(f"\n{'=' * 70}\nAll {len(SCRIPTS)} compute scripts finished successfully.\n{'=' * 70}")
