import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def create_dataset(series: np.ndarray, window: int, target_col: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window over `series` and build (X, y) pairs."""
    X, y = [], []
    for idx in range(len(series) - window):
        X.append(series[idx : idx + window])
        y.append(series[idx + window, target_col])
    return np.array(X), np.array(y)


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


def run_regular(
    trace_path: Path,
    window: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    output_dir: Path,
) -> None:
    df = pd.read_csv(trace_path)
    df["Time"] = pd.to_timedelta(df["Time"], unit="ms")
    df_reg = df.set_index("Time").resample("1ms").mean(numeric_only=True)
    df_reg["Length"] = df_reg["Length"].interpolate().fillna(method="bfill").fillna(method="ffill")

    scaler = MinMaxScaler()
    series = scaler.fit_transform(df_reg[["Length"]].values)

    X_np, y_np = create_dataset(series, window=window, target_col=0)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)
    train_model(model, loader, epochs=epochs, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy()
    y_true = y_tensor.numpy()

    preds = scaler.inverse_transform(preds)
    y_true = scaler.inverse_transform(y_true)

    time_axis = (df_reg.index[window:].total_seconds() * 1e3).astype(float)
    title = f"Regular - {trace_path.stem} - {epochs} epochs"
    output_path = output_dir / f"{trace_path.stem}_epochs{epochs}_regular.png"
    plot_predictions(time_axis, y_true.squeeze(), preds.squeeze(), title, output_path, is_irregular=False)


def run_irregular(
    trace_path: Path,
    window: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    hidden_dim: int,
    num_layers: int,
    output_dir: Path,
) -> None:
    df = pd.read_csv(trace_path)
    df["delta_t"] = df["Time"].diff().fillna(0.0)

    scaler = MinMaxScaler()
    series = scaler.fit_transform(df[["delta_t", "Length"]].values)

    X_np, y_np = create_dataset(series, window=window, target_col=1)
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)

    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
    train_model(model, loader, epochs=epochs, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy()

    y_true = y_tensor.numpy()

    preds_full = np.zeros((len(preds), 2), dtype=float)
    preds_full[:, 1] = preds[:, 0]
    y_true_full = np.zeros((len(y_true), 2), dtype=float)
    y_true_full[:, 1] = y_true[:, 0]

    preds = scaler.inverse_transform(preds_full)[:, 1]
    y_true = scaler.inverse_transform(y_true_full)[:, 1]

    time_axis = df["Time"].values[window:].astype(float)
    title = f"Irregular - {trace_path.stem} - {epochs} epochs"
    output_path = output_dir / f"{trace_path.stem}_epochs{epochs}_irregular.png"
    plot_predictions(time_axis, y_true, preds, title, output_path, is_irregular=True)


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
