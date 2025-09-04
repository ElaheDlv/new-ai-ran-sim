# ---------------------------
# Simulation Configuration
# ---------------------------
import os
SIM_STEP_TIME_DEFAULT = 1
SIM_HANDOVER_HISTORY_LENGTH = 3
SIM_MAX_STEP = 20000
SIM_SPAWN_UE_AFTER_LOAD_HISTORY_STABLIZED = True

# When True, only real offered traffic (traces/AI-service) results in throughput.
# UEs without offered load will report 0 Mbps (no fallback to achievable rate).
SIM_TRAFFIC_STRICT_REAL_ONLY = os.getenv("SIM_TRAFFIC_STRICT_REAL_ONLY", "0") == "1"

# Freeze mobility (UEs don't move); set speed to 0 and targets to current pos.
SIM_FREEZE_MOBILITY = os.getenv("SIM_FREEZE_MOBILITY", "0") == "1"

# Freeze radio (keep SINR/CQI/MCS constant by skipping per-step updates)
SIM_FREEZE_RADIO = os.getenv("SIM_FREEZE_RADIO", "0") == "1"
