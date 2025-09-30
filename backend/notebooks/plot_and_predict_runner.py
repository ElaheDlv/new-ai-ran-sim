import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


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
    scaled = scaler.fit_transform(df[list(feature_cols)].values.astype(float))

    target_idx = feature_cols.index("Length")
    X_np, y_np = create_event_sequences(scaled, window=window, target_idx=target_idx)
    return X_np, y_np, scaler


def inverse_length_transform(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    feature_cols: Sequence[str],
) -> np.ndarray:
    target_idx = feature_cols.index("Length")
    data_min = scaler.data_min_[target_idx]
    data_range = scaler.data_range_[target_idx]
    return scaled_values * data_range + data_min


def run_regular(
    trace_path: Path,
    window: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    output_dir: Path,
    feature_cols: Sequence[str],
) -> None:
    df = pd.read_csv(trace_path)
    df = df.sort_values("Time").reset_index(drop=True)
    df["delta_t"] = df["Time"].diff().fillna(0.0)

    X_np, y_np, scaler = build_scaled_sequences(df, feature_cols=feature_cols, window=window)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers)
    train_model(model, loader, epochs=epochs, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy()
    y_true = y_tensor.numpy()

    preds = inverse_length_transform(preds.squeeze(), scaler, feature_cols)
    y_true = inverse_length_transform(y_true.squeeze(), scaler, feature_cols)

    time_axis = df["Time"].values.astype(float)
    feature_tag = "_".join(feature_cols).lower()
    title = f"Event-Seq ({' + '.join(feature_cols)}) - {trace_path.stem} - {epochs} epochs"
    output_path = output_dir / f"{trace_path.stem}_epochs{epochs}_{feature_tag}.png"
    plot_predictions(time_axis, y_true, preds, title, output_path, is_irregular="delta_t" in feature_cols)


def run_irregular(
    *args,
    **kwargs,
) -> None:
    # Backward compatibility shim – calls run_regular with delta_t included
    return run_regular(*args, feature_cols=("delta_t", "Length"), **kwargs)


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

    run_regular(
        trace_path=args.trace,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dir=output_dir,
        feature_cols=("Length",),
    )

    run_irregular(
        trace_path=args.trace,
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
