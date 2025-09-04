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
