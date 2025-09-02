# ---------------------------
# RAN Configuration
# ---------------------------
RAN_BS_LOAD_HISTORY_LENGTH = 3
RAN_BS_REF_SIGNAL_DEFAULT_TRASNMIT_POWER = 40

# Usually n1 bands are often frequency-division duplex (FDD) bands, while n78 and n258 are typically time-division duplex (TDD) bands.
# so while n78 and n258 bands needs to be split into downlink and uplink separately, n1 bands are usually not split
# as they have a fixed downlink/uplink ratio linked to separate carrier frequencies
# but for simplicity, we use the same split mechanism for all bands.
RAN_CELL_DL_UL_PRB_SPLIT = {
    "n1": (0.7, 0.3),
    "n78": (0.8, 0.2),
    "n258": (0.9, 0.1),
}

RAN_SYMBOL_DURATION = 71.4e-6  # 71.4 µs per OFDM symbol
RAN_SYMBOLS_PER_SLOT = 14
RAN_SLOT_DURATION = RAN_SYMBOL_DURATION * RAN_SYMBOLS_PER_SLOT  # ~1 ms
RAN_SUBCARRIERS_PER_PRB = 12
RAN_SUBCARRIER_SPACING_Hz = 15e3  # 15 kHz
RAN_PRB_BANDWIDTH_Hz = RAN_SUBCARRIERS_PER_PRB * RAN_SUBCARRIER_SPACING_Hz  # 180 kHz
RAN_RESOURCE_ELEMENTS_PER_PRB_PER_SLOT = (
    RAN_SUBCARRIERS_PER_PRB * RAN_SYMBOLS_PER_SLOT
)  # 12 subcarriers * 14 symbols = 168 resource elements per PRB per slot

# Live KPI Dashboard controls
# 0 means unlimited per-UE DL PRB cap; UI will map 0 -> unlimited.
RAN_PRB_CAP_SLIDER_DEFAULT = 0
RAN_PRB_CAP_SLIDER_MAX = 50

# ETSI TS 138 214 V15.3.0. Release 15 Table 5.1.3.1-2 MCS index table 2 for PDSCH
# MCS Index (I_MCS) | Modulation Order Qm | Target code Rate R x [1024] | Spectral efficiency
RAN_MCS_SPECTRAL_EFFICIENCY_TABLE = {
    0: {"modulation_order": 2, "target_code_rate": 120, "spectral_efficiency": 0.2344},
    1: {"modulation_order": 2, "target_code_rate": 193, "spectral_efficiency": 0.3770},
    2: {"modulation_order": 2, "target_code_rate": 308, "spectral_efficiency": 0.6016},
    3: {"modulation_order": 2, "target_code_rate": 449, "spectral_efficiency": 0.8770},
    4: {"modulation_order": 2, "target_code_rate": 602, "spectral_efficiency": 1.1758},
    5: {"modulation_order": 4, "target_code_rate": 378, "spectral_efficiency": 1.4766},
    6: {"modulation_order": 4, "target_code_rate": 434, "spectral_efficiency": 1.6953},
    7: {"modulation_order": 4, "target_code_rate": 490, "spectral_efficiency": 1.9141},
    8: {"modulation_order": 4, "target_code_rate": 553, "spectral_efficiency": 2.1602},
    9: {"modulation_order": 4, "target_code_rate": 616, "spectral_efficiency": 2.4063},
    10: {"modulation_order": 4, "target_code_rate": 658, "spectral_efficiency": 2.5703},
    11: {"modulation_order": 6, "target_code_rate": 466, "spectral_efficiency": 2.7305},
    12: {"modulation_order": 6, "target_code_rate": 517, "spectral_efficiency": 3.0293},
    13: {"modulation_order": 6, "target_code_rate": 567, "spectral_efficiency": 3.3223},
    14: {"modulation_order": 6, "target_code_rate": 616, "spectral_efficiency": 3.6094},
    15: {"modulation_order": 6, "target_code_rate": 666, "spectral_efficiency": 3.9023},
    16: {"modulation_order": 6, "target_code_rate": 719, "spectral_efficiency": 4.2129},
    17: {"modulation_order": 6, "target_code_rate": 772, "spectral_efficiency": 4.5234},
    18: {"modulation_order": 6, "target_code_rate": 822, "spectral_efficiency": 4.8164},
    19: {"modulation_order": 6, "target_code_rate": 873, "spectral_efficiency": 5.1152},
    20: {
        "modulation_order": 8,
        "target_code_rate": 682.5,
        "spectral_efficiency": 5.3320,
    },
    21: {"modulation_order": 8, "target_code_rate": 711, "spectral_efficiency": 5.5547},
    22: {"modulation_order": 8, "target_code_rate": 754, "spectral_efficiency": 5.8906},
    23: {"modulation_order": 8, "target_code_rate": 797, "spectral_efficiency": 6.2266},
    24: {"modulation_order": 8, "target_code_rate": 841, "spectral_efficiency": 6.5703},
    25: {"modulation_order": 8, "target_code_rate": 885, "spectral_efficiency": 6.9141},
    26: {
        "modulation_order": 8,
        "target_code_rate": 916.5,
        "spectral_efficiency": 7.1602,
    },
    27: {"modulation_order": 8, "target_code_rate": 948, "spectral_efficiency": 7.4063},
    # 28: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
    # 29: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
    # 30: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
    # 31: {  # reserved
    #     "modulation_order": 0,
    #     "target_code_rate": 0,
    #     "spectral_efficiency": 0,
    # },
}


def RAN_BS_DEFAULT_CELLS(bs_id):
    return [
        # {
        #     "cell_id": f"{bs_id}_cell_low_freq",
        #     "frequency_band": "n1",
        #     "carrier_frequency_MHz": 2100,
        #     "bandwidth_Hz": 20e6,
        #     "max_prb": 106,
        #     "max_dl_prb": int(RAN_CELL_DL_UL_PRB_SPLIT["n1"][0] * 106),
        #     "max_ul_prb": 106 - int(RAN_CELL_DL_UL_PRB_SPLIT["n1"][0] * 106),
        #     "cell_radius": 1500,
        #     "transmit_power_dBm": 40,
        #     "cell_individual_offset_dBm": 0,
        #     "frequency_priority": 3,
        #     "qrx_level_min": -110,
        # },
        {
            "cell_id": f"{bs_id}_cell_mid_freq",
            "frequency_band": "n78",
            "carrier_frequency_MHz": 3500,
            "bandwidth_Hz": 100e6,
            "max_prb": 273,
            "max_dl_prb": int(RAN_CELL_DL_UL_PRB_SPLIT["n78"][0] * 273),
            "max_ul_prb": 273 - int(RAN_CELL_DL_UL_PRB_SPLIT["n78"][0] * 273),
            "cell_radius": 800,
            "transmit_power_dBm": 40,
            "cell_individual_offset_dBm": 0,
            "frequency_priority": 5,
            "qrx_level_min": -98,
        },
        {
            "cell_id": f"{bs_id}_cell_high_freq",
            "frequency_band": "n258",
            "carrier_frequency_MHz": 26000,
            "bandwidth_Hz": 400e6,
            "max_prb": 264,
            "max_dl_prb": int(RAN_CELL_DL_UL_PRB_SPLIT["n258"][0] * 264),
            "max_ul_prb": 264 - int(RAN_CELL_DL_UL_PRB_SPLIT["n258"][0] * 264),
            "cell_radius": 300,
            "transmit_power_dBm": 50,  # assume achieved by beamforming
            "cell_individual_offset_dBm": 13,
            "frequency_priority": 7,
            "qrx_level_min": -89,
        },
    ]


# Event	Description	Condition
# A1	Serving becomes better than threshold	Serving > threshold
# A2	Serving becomes worse than threshold	Serving < threshold
# A3	Neighbor becomes offset better than serving	Neighbor > Serving + offset
# A4	Neighbor becomes better than threshold	Neighbor > threshold
# A5	Serving becomes worse than threshold1 AND neighbor becomes better than threshold2	Serving < threshold1 AND Neighbor > threshold2
# B1	Inter-RAT neighbor becomes better than threshold	Used for LTE/other RATs
# B2	Serving becomes worse than threshold1 AND Inter-RAT neighbor becomes better than threshold2	Also for Inter-RAT
def RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS():
    return [
        {
            "event_id": "A3",
            "power_threshold": 3,
            "time_to_trigger_in_sim_steps": 3,  # normally time-t-trigger or TTT is in time, but since we are in simulation, we can use simulation steps
        }
    ]


def RAN_BS_EDGE_DEFAULT_SERVER():
    return {
        "node_id": "LAP004262",
        "device_type": "DeviceType.CPU",
        "cpu_memory_GB": 10,  # set this to the docker CPU memory limit minus some overhead.
        "device_memory_GB": 0.0,  # this is the device (GPU, FPGA, etc.) memory, set to 0 if not applicable
    }


RAN_DEFAULT_BS_LIST = [
    {
        "bs_id": "bs_11",
        "position_x": 500,
        "position_y": 500,
        "cell_list": RAN_BS_DEFAULT_CELLS("bs_11"),
        "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
        "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
    },
    {
        "bs_id": "bs_12",
        "position_x": 1500,
        "position_y": 500,
        "cell_list": RAN_BS_DEFAULT_CELLS("bs_12"),
        "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
        "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
    },
    {
        "bs_id": "bs_21",
        "position_x": 500,
        "position_y": 1500,
        "cell_list": RAN_BS_DEFAULT_CELLS("bs_21"),
        "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
        "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
    },
    {
        "bs_id": "bs_22",
        "position_x": 1500,
        "position_y": 1500,
        "cell_list": RAN_BS_DEFAULT_CELLS("bs_22"),
        "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
        "edge_server": RAN_BS_EDGE_DEFAULT_SERVER(),
    },
    # {
    #     "bs_id": "bs_31",
    #     "position_x": 500,
    #     "position_y": 2500,
    #     "cell_list": RAN_BS_DEFAULT_CELLS("bs_31"),
    #     "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
    # },
    # {
    #     "bs_id": "bs_32",
    #     "position_x": 1500,
    #     "position_y": 2500,
    #     "cell_list": RAN_BS_DEFAULT_CELLS("bs_32"),
    #     "rrc_measurement_events": RAN_BS_DEFAULT_RRC_MEASUREMENT_EVENTS(),
    # },
]
