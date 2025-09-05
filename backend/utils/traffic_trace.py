import csv
from typing import List, Tuple, Dict, Optional
import math


def load_csv_trace(path: str) -> List[Tuple[float, int, int]]:
    """
    Load a traffic trace CSV into a list of (t_seconds, dl_bytes, ul_bytes).

    Expected headers (case-sensitive any of):
      - time columns: one of ["t_s", "time", "timestamp"]
      - data columns: "dl_bytes" and optionally "ul_bytes"

    If ul_bytes is missing, it defaults to 0.
    Timestamps are normalized to start at 0.0.
    """
    samples: List[Tuple[float, int, int]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        fields = {name.strip(): name for name in reader.fieldnames}
        # map possible time headers
        t_keys = [k for k in ("t_s", "time", "timestamp") if k in fields]
        if not t_keys:
            # try lowercase
            lowered = {k.lower(): k for k in fields}
            t_keys = [lowered.get(k) for k in ("t_s", "time", "timestamp") if lowered.get(k)]
        if not t_keys:
            raise KeyError("Trace CSV must contain a time column: t_s, time, or timestamp")
        t_key = t_keys[0]
        dl_key = fields.get("dl_bytes")
        ul_key = fields.get("ul_bytes")
        if dl_key is None:
            raise KeyError("Trace CSV must contain 'dl_bytes' column")
        for row in reader:
            if not row:
                continue
            try:
                t = float(row[t_key])
            except Exception:
                continue
            try:
                dl = int(float(row[dl_key]))
            except Exception:
                dl = 0
            ul = 0
            if ul_key is not None:
                try:
                    ul = int(float(row[ul_key]))
                except Exception:
                    ul = 0
            samples.append((t, dl, ul))
    if not samples:
        return []
    # normalize to start at 0
    t0 = samples[0][0]
    norm = [(s[0] - t0, s[1], s[2]) for s in samples]
    return norm


def _find_col(name_map: Dict[str, int], candidates: List[str]) -> int:
    """Find a column index by trying exact, then startswith, then contains matches."""
    for n in candidates:
        if n in name_map:
            return name_map[n]
    for n in candidates:
        for k, idx in name_map.items():
            if k.startswith(n):
                return idx
    for n in candidates:
        for k, idx in name_map.items():
            if n in k:
                return idx
    raise KeyError(f"Columns {candidates} not found; got: {list(name_map.keys())}")


def detect_device_ip(path: str) -> Optional[str]:
    """Heuristically detect the UE/device IP from Source/Destination counts."""
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            name_map: Dict[str, int] = {h.strip().lower(): i for i, h in enumerate(header)}
            idx_src = _find_col(name_map, ["source", "src", "ip.src"])  # type: ignore[arg-type]
            idx_dst = _find_col(name_map, ["destination", "dst", "ip.dst"])  # type: ignore[arg-type]
            counts: Dict[str, int] = {}
            for row in reader:
                if not row:
                    continue
                if len(row) > idx_src:
                    ip = row[idx_src].strip()
                    if ip:
                        counts[ip] = counts.get(ip, 0) + 1
                if len(row) > idx_dst:
                    ip = row[idx_dst].strip()
                    if ip:
                        counts[ip] = counts.get(ip, 0) + 1
            if not counts:
                return None
            return max(counts.items(), key=lambda kv: kv[1])[0]
    except Exception:
        return None


def load_raw_packet_csv(
    path: str,
    ue_ip: Optional[str] = None,
    bin_s: float = 1.0,
    overhead_sub_bytes: int = 70,
) -> List[Tuple[float, int, int]]:
    """
    Load a packet-level CSV (columns: Time, Source, Destination, Length) and
    aggregate into (t_seconds, dl_bytes, ul_bytes) with bin size `bin_s`.

    - `ue_ip` is used to classify direction: DL if Destination==ue_ip; UL if Source==ue_ip
    - Time may be float seconds or any numeric; we floor(Time/bin_s)*bin_s per sample,
      then produce uniform bins starting from the first bin to the last observed.
    """
    # Read header row to find columns (case-insensitive)
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        name_map: Dict[str, int] = {h.strip().lower(): i for i, h in enumerate(header)}

        idx_time = _find_col(name_map, ["t_s", "time", "timestamp", "frame.time_epoch"]) if name_map else 0
        idx_src = _find_col(name_map, ["source", "src", "ip.src"])  # type: ignore[arg-type]
        idx_dst = _find_col(name_map, ["destination", "dst", "ip.dst"])  # type: ignore[arg-type]
        idx_len = _find_col(name_map, ["length", "len", "frame.len", "bytes", "size"])  # type: ignore[arg-type]

        # Aggregate by bins
        bins: Dict[float, Tuple[int, int]] = {}
        first_t = None
        if ue_ip is None:
            ue_ip = detect_device_ip(path)
        for row in reader:
            if not row or len(row) <= max(idx_time, idx_src, idx_dst, idx_len):
                continue
            try:
                t = float(row[idx_time])
                length = int(float(row[idx_len]))
            except Exception:
                continue
            src = row[idx_src].strip()
            dst = row[idx_dst].strip()
            if first_t is None:
                first_t = t
            # Normalize time to start at zero
            t0 = t - (first_t or 0.0)
            # Bin index
            b = math.floor(t0 / max(1e-6, float(bin_s))) * float(bin_s)
            dl, ul = bins.get(b, (0, 0))
            # Adjust length to emulate UDP payload (subtract headers/trailer)
            adj = max(0, int(length) - max(0, int(overhead_sub_bytes)))
            if ue_ip and dst == ue_ip:
                dl += adj
            if ue_ip and src == ue_ip:
                ul += adj
            bins[b] = (dl, ul)

        if not bins:
            return []
        # Produce ordered list of (t, dl, ul)
        ts = sorted(bins.keys())
        samples: List[Tuple[float, int, int]] = [(t, bins[t][0], bins[t][1]) for t in ts]
        return samples
