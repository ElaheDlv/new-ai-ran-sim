import csv
import os
import logging
from typing import List, Tuple, Dict, Optional, Any
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
    overhead_sub_bytes: int = 0,
) -> List[Tuple[float, int, int]]:
    """
    Parse a raw packet capture exported as CSV and aggregate it into
    evenly binned traffic samples suitable for replay.

    Input CSV is expected to have time and IP endpoints (case-insensitive
    column names are supported). Common headers include:
      - Time: one of [t_s, time, timestamp, frame.time_epoch]
      - Source/Destination: [source/src/ip.src] and [destination/dst/ip.dst]
      - Length: one of [length, len, frame.len, bytes, size]

    Aggregation rules:
      - Each packet contributes (length - overhead_sub_bytes) bytes.
      - Direction is classified by `ue_ip`:
          DL if Destination == ue_ip; UL if Source == ue_ip.
        If `ue_ip` is None, it is detected heuristically from endpoint counts.
      - Time is normalized to start at 0 using the first observed timestamp.
      - Packets are grouped into bins of width `bin_s` seconds using
        floor((t - t0)/bin_s) * bin_s; the output is a list of
        (bin_time_s, dl_bytes, ul_bytes) sorted by time.
    """
    # Read header row to build a name->index map (case-insensitive)
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        # Lowercase keys for case-insensitive lookups
        name_map: Dict[str, int] = {h.strip().lower(): i for i, h in enumerate(header)}

        # Locate required columns with tolerant matching
        idx_time = _find_col(name_map, ["t_s", "time", "timestamp", "frame.time_epoch"]) if name_map else 0
        idx_src = _find_col(name_map, ["source", "src", "ip.src"])  # type: ignore[arg-type]
        idx_dst = _find_col(name_map, ["destination", "dst", "ip.dst"])  # type: ignore[arg-type]
        idx_len = _find_col(name_map, ["length", "len", "frame.len", "bytes", "size"])  # type: ignore[arg-type]

        # Aggregate packet lengths into (DL, UL) buckets keyed by bin timestamp
        bins: Dict[float, Tuple[int, int]] = {}
        first_t = None
        # If UE IP is not provided, try to auto-detect it from endpoint counts
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
            # Normalize time to start at zero (relative to first packet)
            t0 = t - (first_t or 0.0)
            # Compute bin start time with numeric stability guards
            b = math.floor(t0 / max(1e-6, float(bin_s))) * float(bin_s)
            dl, ul = bins.get(b, (0, 0))
            # Subtract protocol overhead to approximate payload bytes
            adj = max(0, int(length) - max(0, int(overhead_sub_bytes)))
            if ue_ip and dst == ue_ip:
                dl += adj
            if ue_ip and src == ue_ip:
                ul += adj
            bins[b] = (dl, ul)

        if not bins:
            return []
        # Produce ordered list of (t, dl, ul) sorted by bin time
        ts = sorted(bins.keys())
        samples: List[Tuple[float, int, int]] = [(t, bins[t][0], bins[t][1]) for t in ts]
        return samples


# -------------------------------------------------------------
# Trace validation helpers
# -------------------------------------------------------------

def _first_n(reader, n: int) -> List[Any]:
    out = []
    for i, row in enumerate(reader):
        if i >= n:
            break
        out.append(row)
    return out


def validate_preaggregated_trace_csv(path: str, sample_rows: int = 10) -> Dict[str, Any]:
    """Validate that a pre-aggregated CSV exists and has required columns.

    Returns a dict with keys: exists, valid, error, columns, sample_count.
    """
    res: Dict[str, Any] = {
        "kind": "agg",
        "path": path,
        "exists": os.path.exists(path),
        "valid": False,
        "error": None,
        "columns": {},
        "sample_count": 0,
    }
    if not res["exists"]:
        res["error"] = "File not found"
        return res

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                res["error"] = "CSV has no header"
                return res
            fields = {name.strip(): name for name in reader.fieldnames}
            t_keys = [k for k in ("t_s", "time", "timestamp") if k in fields]
            dl_key = fields.get("dl_bytes")
            ul_key = fields.get("ul_bytes")
            if not t_keys:
                res["error"] = "Missing time column (t_s/time/timestamp)"
                return res
            if dl_key is None:
                res["error"] = "Missing 'dl_bytes' column"
                return res
            res["columns"] = {"time": t_keys[0], "dl_bytes": dl_key, "ul_bytes": ul_key}

            rows = _first_n(reader, sample_rows)
            ok_rows = 0
            for row in rows:
                try:
                    float(row[t_keys[0]])
                    float(row[dl_key])
                    ok_rows += 1
                except Exception:
                    continue
            res["sample_count"] = ok_rows
            if ok_rows == 0:
                res["error"] = "No parsable rows in first sample"
                return res
            res["valid"] = True
            return res
    except Exception as e:
        res["error"] = str(e)
        return res


def validate_raw_packet_trace_csv(path: str, sample_rows: int = 100) -> Dict[str, Any]:
    """Validate that a raw packet CSV exists and has required columns.

    Returns a dict with keys: exists, valid, error, columns, detected_ue_ip, sample_count.
    """
    res: Dict[str, Any] = {
        "kind": "raw",
        "path": path,
        "exists": os.path.exists(path),
        "valid": False,
        "error": None,
        "columns": {},
        "detected_ue_ip": None,
        "sample_count": 0,
    }
    if not res["exists"]:
        res["error"] = "File not found"
        return res

    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                res["error"] = "Empty CSV"
                return res
            name_map: Dict[str, int] = {h.strip().lower(): i for i, h in enumerate(header)}
            try:
                idx_time = _find_col(name_map, ["t_s", "time", "timestamp", "frame.time_epoch"])  # type: ignore[arg-type]
                idx_src = _find_col(name_map, ["source", "src", "ip.src"])  # type: ignore[arg-type]
                idx_dst = _find_col(name_map, ["destination", "dst", "ip.dst"])  # type: ignore[arg-type]
                idx_len = _find_col(name_map, ["length", "len", "frame.len", "bytes", "size"])  # type: ignore[arg-type]
            except KeyError as e:
                res["error"] = str(e)
                return res
            res["columns"] = {
                "time": list(name_map.keys())[idx_time] if idx_time is not None else None,
                "src": list(name_map.keys())[idx_src] if idx_src is not None else None,
                "dst": list(name_map.keys())[idx_dst] if idx_dst is not None else None,
                "len": list(name_map.keys())[idx_len] if idx_len is not None else None,
            }

            rows = _first_n(reader, sample_rows)
            ok_rows = 0
            for row in rows:
                if not row or len(row) <= max(idx_time, idx_src, idx_dst, idx_len):
                    continue
                try:
                    float(row[idx_time])
                    float(row[idx_len])
                except Exception:
                    continue
                ok_rows += 1
            res["sample_count"] = ok_rows
            if ok_rows == 0:
                res["error"] = "No parsable rows in first sample"
                return res
            res["detected_ue_ip"] = detect_device_ip(path)
            res["valid"] = True
            return res
    except Exception as e:
        res["error"] = str(e)
        return res


def validate_traces_configuration(
    trace_map: Optional[Dict[str, str]] = None,
    raw_map: Optional[List[Dict[str, Optional[str]]]] = None,
    bin_s: float = 1.0,
    overhead_bytes: int = 70,
    logger_name: str = __name__,
) -> Dict[str, Any]:
    """Validate configured traces and log a concise summary.

    Returns a dict with 'all_valid' flag and per-file results.
    """
    logger = logging.getLogger(logger_name)
    results: Dict[str, Any] = {"all_valid": True, "agg": {}, "raw": {}}

    # ANSI colors for visibility (safe to display in most terminals)
    C_RESET = "\033[0m"
    C_GREEN = "\033[92m"
    C_RED = "\033[91m"
    C_CYAN = "\033[96m"
    C_YELLOW = "\033[93m"
    C_BOLD = "\033[1m"

    logger.info(f"{C_CYAN}{C_BOLD}=== TRACE VALIDATION START ==={C_RESET}")

    trace_map = trace_map or {}
    for imsi, path in trace_map.items():
        r = validate_preaggregated_trace_csv(path)
        results["agg"][imsi] = r
        if not r.get("valid"):
            results["all_valid"] = False
            logger.error(f"{C_RED}Trace (agg) for {imsi}: {path} -> INVALID: {r.get('error')}{C_RESET}")
        else:
            logger.info(f"{C_GREEN}Trace (agg) for {imsi}: {path} -> OK{C_RESET} "
                        f"(cols={r['columns']}, sample_rows={r['sample_count']})")

    raw_map = raw_map or []
    for item in raw_map:
        if not isinstance(item, dict):
            continue
        imsi = item.get("imsi") or "?"
        path = item.get("file") or ""
        r = validate_raw_packet_trace_csv(path)
        results["raw"][imsi] = r
        if not r.get("valid"):
            results["all_valid"] = False
            logger.error(f"{C_RED}Trace (raw) for {imsi}: {path} -> INVALID: {r.get('error')}{C_RESET}")
        else:
            ip_note = f", ue_ip_auto={r['detected_ue_ip']}" if r.get("detected_ue_ip") else ""
            logger.info(f"{C_GREEN}Trace (raw) for {imsi}: {path} -> OK{C_RESET} "
                        f"(cols={r['columns']}, sample_rows={r['sample_count']}{ip_note})")

    if not results["all_valid"]:
        logger.warning(f"{C_YELLOW}Some traces invalid. Simulation will continue but affected UEs may have no replayed traffic.{C_RESET}")
    else:
        logger.info(f"{C_GREEN}All configured traces look valid.{C_RESET}")
    logger.info(f"{C_CYAN}{C_BOLD}=== TRACE VALIDATION END ==={C_RESET}")
    return results
