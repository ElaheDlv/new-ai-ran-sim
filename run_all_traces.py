import subprocess
from pathlib import Path

SCRIPT = "backend/notebooks/prev_plot_and_predict_runner.py"
TRACE_DIR = Path("backend/assets/traces")
ARGS = [
    "--epochs", "1500",
    "--window", "128",
    "--output-dir", "backend/assets/plots",
    "--feature-sets", "length", "delta_t+length", "uniform-length",
    "--early-stop", "10",
    "--val-ratio", "0.1",
    "--batch-size", "64",
]

def main():
    for trace in sorted(TRACE_DIR.glob("*.csv")):
        print(f"\n=== Running on {trace.name} ===")
        cmd = ["python", SCRIPT, str(trace)] + ARGS
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
