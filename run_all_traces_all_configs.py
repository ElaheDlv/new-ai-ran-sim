import subprocess
from pathlib import Path
import shlex   # NEW



SCRIPT = "backend/notebooks/prev_plot_and_predict_runner.py"
TRACE_DIR = Path("backend/assets/traces")
CONFIG_FILE = Path("lstm_configs.txt")   # NEW
#CONFIG_FILE = Path("mega_sweep_configs.txt") 

BASE_ARGS = [
    "--epochs", "1500",
    "--window", "128",
    "--output-dir", "backend/assets/plots/lstm_configs_seperate",  # UPDATED
    "--feature-sets", "length", "delta_t+length", "uniform-length",
    "--early-stop", "10",
    "--val-ratio", "0.1",
    "--batch-size", "64",
]


def load_configs():
    configs = []
    with open(CONFIG_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):  # skip empty or commented lines
                configs.append(shlex.split(line))
    return configs

def main():
    configs = load_configs()
    for trace in sorted(TRACE_DIR.glob("*.csv")):
        for cfg in configs:
            print(f"\n=== Running on {trace.name} with config: {' '.join(cfg)} ===")
            cmd = ["python", SCRIPT, str(trace)] + BASE_ARGS + cfg
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
