import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


FEATURE_SETS: Dict[str, Dict[str, Sequence[str]]] = {
    "length": {"columns": ("Length",), "mode": "event"},
    "delta_t+length": {"columns": ("delta_t", "Length"), "mode": "event"},
    "uniform-length": {"columns": ("Length",), "mode": "uniform"},
}


def create_event_sequences(
    series: np.ndarray,
    window: int,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build padded event-history windows without resampling."""

    feature_dim = series.shape[1]
    X, y = [], []

    for idx in range(series.shape[0]):
        start = max(0, idx - window)
        history = series[start:idx]
        if history.shape[0] < window:
            pad = np.zeros((window - history.shape[0], feature_dim), dtype=series.dtype)
            history = np.vstack((pad, history))
        X.append(history)
        y.append(series[idx, target_idx])

    return np.stack(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take final time step
        return out


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(loader))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")


def plot_predictions(
    time_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    output_path: Path,
    is_irregular: bool,
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, y_true, label="True")
    plt.plot(time_axis, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time [ms]" if not is_irregular else "Time [ms] (irregular sampling)")
    plt.ylabel("Length [bytes]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot -> {output_path}")


def build_scaled_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale selected columns and build padded sequences."""

    scaler = MinMaxScaler()
    values = df[list(feature_cols)].values.astype(np.float32)
    scaled = scaler.fit_transform(values).astype(np.float32)

    target_idx = feature_cols.index("Length")
    X_np, y_np = create_event_sequences(scaled, window=window, target_idx=target_idx)
    return X_np.astype(np.float32), y_np.astype(np.float32), scaler


def inverse_length_transform(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    feature_cols: Sequence[str],
) -> np.ndarray:
    target_idx = feature_cols.index("Length")
    data_min = scaler.data_min_[target_idx]
    data_range = scaler.data_range_[target_idx]
    return scaled_values * data_range + data_min


def predict_in_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
            targets.append(yb.cpu())
    pred_cat = torch.cat(preds, dim=0)
    target_cat = torch.cat(targets, dim=0)
    return pred_cat.numpy().squeeze(-1), target_cat.numpy().squeeze(-1)


def train_feature_set(
    df_event: pd.DataFrame,
    trace_path: Path,
    feature_cols: Sequence[str],
    feature_tag: str,
    mode: str,
    window: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    output_dir: Path,
) -> None:
    if "Length" not in feature_cols:
        raise ValueError("Feature set must include 'Length'.")

    if mode == "uniform":
        df_prepared = build_uniform_dataframe(df_event)
    else:
        df_prepared = df_event

    X_np, y_np, scaler = build_scaled_sequences(df_prepared, feature_cols=feature_cols, window=window)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers)
    train_model(model, train_loader, epochs=epochs, device=device)

    preds, y_true = predict_in_batches(model, eval_loader, device=device)

    preds = inverse_length_transform(preds, scaler, feature_cols)
    y_true = inverse_length_transform(y_true, scaler, feature_cols)

    time_axis = df_prepared["Time"].values.astype(float)
    title = f"Seq ({' + '.join(feature_cols)}) [{mode}] - {trace_path.stem} - {epochs} epochs"
    output_path = output_dir / f"{trace_path.stem}_epochs{epochs}_{feature_tag}.png"
    plot_predictions(time_axis, y_true, preds, title, output_path, is_irregular="delta_t" in feature_cols)


def load_trace(trace_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trace_path)
    df = df.sort_values("Time").reset_index(drop=True)
    if "delta_t" not in df.columns:
        df["delta_t"] = df["Time"].diff().fillna(0.0)
    return df


def build_uniform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the event stream onto a uniform grid based on the smallest Δt."""

    if df.empty:
        return df.copy()

    time_vals = df["Time"].astype(float).to_numpy()
    deltas = np.diff(time_vals)
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.size == 0:
        min_gap = 1.0
    else:
        min_gap = positive_deltas.min()

    if min_gap <= 0:
        min_gap = 1.0

    start = float(time_vals[0])
    stop = float(time_vals[-1])
    total_span = max(stop - start, min_gap)
    max_steps = 2_000_000

    steps = int(np.floor(total_span / min_gap)) + 1
    if steps > max_steps:
        min_gap = total_span / max_steps
        steps = max_steps + 1

    grid = start + np.arange(steps) * min_gap
    length_series = np.zeros_like(grid)

    for t, length in zip(time_vals, df["Length"].to_numpy(dtype=float)):
        idx = int(round((t - start) / min_gap))
        if 0 <= idx < steps:
            length_series[idx] += length

    uniform_df = pd.DataFrame({"Time": grid, "Length": length_series})
    uniform_df["delta_t"] = min_gap
    return uniform_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM forecasters on telecom traces.")
    parser.add_argument("trace", type=Path, help="Path to the CSV trace file.")
    parser.add_argument("--window", type=int, default=20, help="Sliding window size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for each model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension of the LSTM.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device to use. Default picks CUDA if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["length", "delta_t+length"],
        choices=sorted(FEATURE_SETS.keys()),
        help="One or more feature sets to train (default: length and delta_t+length).",
    )
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "cuda":
        return torch.device("cuda")
    if arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df_event = load_trace(args.trace)

    for feature_name in args.feature_sets:
        config = FEATURE_SETS[feature_name]
        cols = config["columns"]
        mode = config.get("mode", "event")
        print(f"Training feature set '{feature_name}' ({mode}) -> {cols}")
        train_feature_set(
            df_event=df_event,
            trace_path=args.trace,
            feature_cols=cols,
            feature_tag=feature_name.replace("+", "_").replace(" ", ""),
            mode=mode,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()