import subprocess
from pathlib import Path

SCRIPT = "backend/notebooks/plot_and_predict_runner.py"
TRACE_DIR = Path("backend/assets/traces")
ARGS = [
    "--epochs", "15",
    "--window", "20",
    "--output-dir", "backend/assets/plots",
    "--feature-sets", "length", "delta_t+length", "uniform-length",
]

def main():
    for trace in sorted(TRACE_DIR.glob("*.csv")):
        print(f"\n=== Running on {trace.name} ===")
        cmd = ["python", SCRIPT, str(trace)] + ARGS
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
