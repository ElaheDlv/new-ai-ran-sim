# ---------------------------
# Simulation Configuration
# ---------------------------
import os
SIM_STEP_TIME_DEFAULT = 1
SIM_HANDOVER_HISTORY_LENGTH = 3
SIM_MAX_STEP = 20000
SIM_SPAWN_UE_AFTER_LOAD_HISTORY_STABLIZED = True


# Freeze mobility (UEs don't move); set speed to 0 and targets to current pos.
SIM_FREEZE_MOBILITY = os.getenv("SIM_FREEZE_MOBILITY", "0") == "1"

# ---------------------------
# Real-traffic trace replay
# ---------------------------
import json as _json

# Time scaling for trace playback (1.0 = realtime)
try:
    TRACE_SPEEDUP = float(os.getenv("TRACE_SPEEDUP", "1.0"))
except Exception:
    TRACE_SPEEDUP = 1.0

# When true, only show served traffic from trace buffers (no fallback)
STRICT_REAL_TRAFFIC = os.getenv("STRICT_REAL_TRAFFIC", "0") == "1"

# Raw packet CSV mapping (repeatable via CLI)
_trace_raw_env = os.getenv("TRACE_RAW_MAP_JSON", "")
try:
    _raw = _json.loads(_trace_raw_env) if _trace_raw_env else []
    if isinstance(_raw, dict):
        # Convert dict {imsi: {file, ue_ip}} or {imsi: "file:ue_ip"} to list
        TRACE_RAW_MAP = []
        for k, v in _raw.items():
            if isinstance(v, dict):
                TRACE_RAW_MAP.append({"imsi": k, "file": v.get("file"), "ue_ip": v.get("ue_ip")})
            elif isinstance(v, str):
                parts = v.split(":")
                TRACE_RAW_MAP.append({"imsi": k, "file": parts[0], "ue_ip": parts[1] if len(parts) > 1 else None})
            else:
                continue
    elif isinstance(_raw, list):
        # Normalize list entries
        norm = []
        for item in _raw:
            if not isinstance(item, dict):
                continue
            imsi = item.get("imsi")
            file = item.get("file")
            ue_ip = item.get("ue_ip")
            if imsi and file:
                norm.append({"imsi": imsi, "file": file, "ue_ip": ue_ip})
        TRACE_RAW_MAP = norm
    else:
        TRACE_RAW_MAP = []
except Exception:
    TRACE_RAW_MAP = []

try:
    TRACE_BIN = float(os.getenv("TRACE_BIN", "1.0"))
except Exception:
    TRACE_BIN = 1.0

# Subtract per-packet overhead when aggregating raw traces (e.g., headers/trailer)
# Default is 0 since the simulator does not re-encapsulate packets; adjust only if
# you want to approximate payload bytes (e.g., 42 for IPv4+UDP, 54 for IPv4+TCP).
try:
    TRACE_OVERHEAD_BYTES = int(os.getenv("TRACE_OVERHEAD_BYTES", "0"))
except Exception:
    TRACE_OVERHEAD_BYTES = 0

# Trace debug controls
TRACE_DEBUG = os.getenv("TRACE_DEBUG", "0") in ("1", "true", "True")
TRACE_DEBUG_IMSI = set([
    s.strip() for s in os.getenv("TRACE_DEBUG_IMSI", "").split(",") if s.strip()
])
