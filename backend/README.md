# AI-RAN Simulator Backend

The **AI-RAN Simulator Backend** is a Python-based simulation engine designed to model and analyze the behavior of 5G Radio Access Networks (RAN). It supports advanced features such as network slicing, mobility management, and intelligent control via xApps. This backend is part of a larger project that includes a frontend for visualization and interaction.

## 📁 Project Structure

backend/
├── main.py # Entry point for the WebSocket server
├── utils/ # Utility functions and classes
├── settings/ # Configuration files for the simulation
├── network_layer/ # network simulation logic
├── knowledge_layer/ # knowledge base, offering explanations for everything in the network layer
├── intelligence_layer/ # user-engaging and decision-making agents

---

## 📦 Requirements

- Python 3.12 or higher
- docker (to deploy the AI services)
- Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

1. Start the WebSocket Server <br>Run the backend server to enable communication with the frontend:

   ```bash
   python main.py
   ```

2. Start the frontend <br>

   ```bash
   cd frontend
   npm run dev
   ```

---

## 🧪 Simple Topology (1 BS • 1 Cell • ~10 UEs)

For quick experiments and debugging, a simple preset is available. It reduces the network to one base station with a single n78 cell and spawns about 10 UEs.

Two ways to enable it:

- CLI flags (recommended)

  - Server mode (WebSocket server, controlled by the UI/client):

    ```bash
    python main.py --preset simple --ue-max 10 --mode server
    ```

  - Headless mode (no WebSocket, runs a short simulation loop and starts xApps like the KPI dashboard):

    ```bash
    python main.py --preset simple --ue-max 10 --mode headless --steps 120
    # Open the KPI dashboard at http://localhost:8061
    ```

- Environment variables (alternative)

  - Create a `.env` in `backend/` or export in shell:

    ```bash
    export RAN_TOPOLOGY_PRESET=simple
    export UE_DEFAULT_MAX_COUNT=10
    python main.py
    ```

What the preset does:

- Base stations: 1 (`bs_1`), see `settings/ran_config.py`.
- Cells: 1 n78 cell attached to `bs_1`.
- UE caps: spawn 1–2 per step, max ≈ 10 (overridable by `--ue-max` or `UE_DEFAULT_MAX_COUNT`).

Return to the full 4‑BS/8‑cell topology by omitting `--preset` (or setting `--preset default`).

### Control how many UEs per slice (simple preset)

You can explicitly choose how many UEs are subscribed to each slice when using the simple preset. These UEs attach to their single subscribed slice deterministically.

- With CLI flags:

```bash
python main.py --preset simple --ue-max 10 \
  --ue-embb 6 --ue-urllc 3 --ue-mmtc 1 
```

- With environment variables:

```bash
export RAN_TOPOLOGY_PRESET=simple
export UE_DEFAULT_MAX_COUNT=10
export UE_SIMPLE_COUNT_EMBB=6
export UE_SIMPLE_COUNT_URLLC=3
export UE_SIMPLE_COUNT_MMTC=1
python main.py
```

Notes:
- Total UEs = sum of the slice counts when `--preset simple`. If you also pass `--ue-max` and it differs, the backend adjusts the total to match the sum of slice counts.
- Omit the slice counts to keep the default randomized distribution.
- Runtime spawn is still dynamic (1–2 per step in simple mode); slice membership is fixed per IMSI.

---


---

## 🧪 Isolate PRB Effects (Freeze Mobility)

If you want KPI changes to come only from PRB allocation (and not from UE movement changing SINR/CQI/MCS), freeze mobility so UEs stay stationary.

Enable via CLI flag or environment variable (works in both server and headless modes):

```bash
# From backend/
export SIM_FREEZE_MOBILITY=1
python main.py --preset simple --mode server \
  --ue-max 3 --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --freeze-mobility

```

What this does:

- Sets all UE speeds to 0 at creation/registration and pins their targets to current positions.
- With positions fixed, radio KPIs (SINR/CQI/MCS) stay constant. DL Mbps then changes only when you adjust PRB allocation (slice shares/Move‑RB/per‑UE cap) or the offered load (traces/AI services).

Tip: You usually don’t need to freeze radio; freezing mobility is sufficient in this simulator to keep the radio constant.

---

## 📈 Replay Raw CSV Traces (per‑UE offered load)

Attach a raw packet CSV so its offered traffic is replayed and served subject to radio capacity and PRB allocation.

Downlink replay and buffering happen at the gNB (Base Station): the BS owns a per‑UE DL queue (bytes) and a per‑UE DL trace replayer (samples, clock, idx). Each simulation step, the BS advances replay clocks and enqueues due DL bytes; cells then serve from the BS queue up to capacity. The UE’s `dl_buffer_bytes` is updated as a mirror for the UI (it reflects the gNB queue).

- CLI flags:
  - `--trace-speedup <x>` (scale time; default 1.0)
  - `--strict-real-traffic` (show only served traffic; no fallback capacity)
  - `--trace-raw-map IMSI_#:path/to/raw.csv:UE_IP` (Wireshark/PCAP CSV; UE_IP required to classify DL/UL)
    You can also attach by slice or ALL UEs:
    - `--trace-raw-map slice:eMBB:path/to/embb.csv:UE_IP`
    - `--trace-raw-map ALL:path/to/trace.csv:UE_IP`
  - `--trace-bin <seconds>` (aggregation bin for raw CSV; default 1.0)
  - `--trace-overhead-bytes <n>` (subtract per-packet bytes in raw CSV; default 0)
  - `--trace-loop` (replay traces continuously)

Key trace flags (what they do):

- `--trace-speedup <x>`: scales the replay clock. 1.0 = real time. 2.0 replays twice as fast (the same traced seconds happen in half the wall-clock time); 0.5 replays at half speed. Affects when DL samples are enqueued into the gNB DL queue; serving still happens per simulation step.
- `--trace-bin <seconds>`: aggregation window for raw packet CSVs. Packets are grouped by `floor((t - t0)/bin)*bin` and summed to produce `(t, dl_bytes, ul_bytes)` samples. Smaller bins (e.g., 0.2) preserve burstiness; larger bins (e.g., 2.0) smooth traffic. Default 1.0 aligns with the simulator’s 1 s step.
- `--trace-loop`: when enabled, traces repeat seamlessly after the last sample. Without this, each trace plays once and stops offering new bytes after the end.

Examples:

```bash
# Using raw packet CSVs (Wireshark export) — headless
python backend/main.py --preset simple --mode headless --steps 180 \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Using raw packet CSVs (Wireshark export) — server
python backend/main.py --preset simple --mode server \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs (raw traces only), headless (one per slice)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility \
  --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, eMBB-only (all use eMBB trace)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility --ue-embb 3 --ue-urllc 0 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, eMBB-only — server
python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 3 --ue-urllc 0 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, URLLC-only (all use URLLC trace)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility --ue-embb 0 --ue-urllc 3 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, URLLC-only — server
python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 0 --ue-urllc 3 --ue-mmtc 0 \
  --trace-raw-map IMSI_0:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, mMTC-only (all use mMTC trace)
python backend/main.py --preset simple --mode headless --steps 180 \
  --freeze-mobility --ue-embb 0 --ue-urllc 0 --ue-mmtc 3 \
  --trace-raw-map IMSI_0:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs, mMTC-only — server
python backend/main.py --preset simple --mode server \
  --freeze-mobility --ue-embb 0 --ue-urllc 0 --ue-mmtc 3 \
  --trace-raw-map IMSI_0:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic

# Three stationary UEs (raw traces only), server mode (one per slice)
python backend/main.py --preset simple --mode server \
  --freeze-mobility \
  --ue-embb 1 --ue-urllc 1 --ue-mmtc 1 \
  --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
  --trace-raw-map IMSI_2:backend/assets/traces/mmtc_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic


  # Three stationary UE (raw traces only), server mode (one per slice) use of mixed file seperated into 3 different files

  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/eMBB.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/URLLC.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/mMTC.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


# Three generated test data for three UE

  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/embb_gen.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/urllc_gen.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/mmtc_gen.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


# Three generated test data for three UE
  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/synthetic_embb.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/synthetic_urllc.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/synthetic_mmtc.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


# Three generated test data for three UE with queuing 
  python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/synthetic_embb_queueing.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/synthetic_urllc_queueing.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/synthetic_mmtc_queueing.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


python backend/main.py --preset simple --mode server   --freeze-mobility   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_0:backend/assets/traces/eMBB_aligned.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/URLLC_aligned.csv:172.30.1.1   --trace-raw-map IMSI_2:backend/assets/traces/mMTC_aligned.csv:172.30.1.1   --trace-bin 1.0 --trace-overhead-bytes 0 --trace-speedup 1.0 --strict-real-traffic --trace-loop


```

How it works:
- Each UE with a trace enqueues `dl_bytes` into a per‑UE buffer at the traced times (scaled by `--trace-speedup`).
- Cells compute capacity from MCS×PRBs and serve from the buffer up to that capacity each step.
- With `--strict-real-traffic`, UE DL Mbps equals served traffic; otherwise, empty buffers show achievable capacity.

Notes:
- Traces are attached by IMSI on spawn/registration. Ensure those IMSIs exist during the run (use `--ue-max`/slice counts with simple preset).
- Place CSVs anywhere; `backend/assets/traces/` is a convenient location.
- Headless mode runs for exactly `--steps` iterations and then exits. Use server mode for an open‑ended run controlled from the frontend (http://localhost:3000) or a WebSocket client.

---

### Attach Traces to All UEs or by Slice (many UEs)

##### This part is still not working properly

When running with many UEs, you can attach traces without listing every IMSI individually:

- Same trace for all UEs (wildcard):

```bash
python backend/main.py --preset simple --mode server \
  --ue-max 30 --freeze-mobility \
  --trace-raw-map ALL:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop
```

- Different traces per slice (applies to any number of UEs in each slice):

```bash
python backend/main.py --preset simple --mode server \
  --ue-embb 10 --ue-urllc 10 --ue-mmtc 10 --freeze-mobility \
  --trace-raw-map slice:eMBB:backend/assets/traces/eMBB_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:URLLC:backend/assets/traces/URLLC_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:mMTC:backend/assets/traces/mMTC_aligned.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop
```

Semantics:
- `IMSI_#:` maps a specific UE.
- `ALL:` applies to any UE not matched by a specific mapping.
- `slice:<NAME>:` applies to UEs registered on that slice (`eMBB`, `URLLC`, `mMTC`).
  Priority: exact IMSI > slice mapping > ALL.

Notes:
- PRB allocation already covers all UEs in each cell; with the above, every UE will replay its trace and compete for PRBs.
- Freezing mobility keeps radio stable so differences come from offered load and PRB allocation.

### RL PRB Allocation + Traces (many UEs)

You can combine the DQN PRB allocator xApp with the multi‑UE trace mapping above to learn PRB shifts under realistic traffic. Install PyTorch first:

```bash
pip install torch
```

- Per‑slice traces with RL (any number of UEs per slice):

```bash
python backend/main.py --preset simple --mode server \
  --ue-embb 10 --ue-urllc 10 --ue-mmtc 10 --freeze-mobility \
  --trace-raw-map slice:eMBB:backend/assets/traces/eMBB_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:URLLC:backend/assets/traces/URLLC_aligned.csv:172.30.1.1 \
  --trace-raw-map slice:mMTC:backend/assets/traces/mMTC_aligned.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

- Same trace for all UEs with RL:

```bash
python backend/main.py --preset simple --mode server \
  --ue-max 30 --freeze-mobility \
  --trace-raw-map ALL:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

Tips and knobs:
- `--dqn-period N` acts every N sim steps (default 1). With the default 1 s step, `N=1` roughly corresponds to T=1 s; adjust if you want a different control period.
- `--dqn-move-step K` moves K PRBs per action (Table 3 uses 1 RB).
- A pre‑trained model can be pointed to with `--dqn-model backend/models/dqn_prb.pt` (this is also the default path); omit `--dqn-train` to run inference only.
- Use the KPI dashboard with history (`--kpi-history --kpi-log`) to monitor slice PRBs, DL Mbps, buffers, and the effect of PRB moves.


## 📊 Live KPI Dashboard xApp

Drop-in xApp that starts a Dash server at `http://localhost:8061` and streams per‑UE and per‑cell KPIs.

- Per‑UE: bitrate (Mbps), SINR, CQI, allocated PRBs.
- Per‑cell: load, PRB usage; fixed PRB quotas per slice (eMBB/URLLC/mMTC).
- Controls:
  - `Max DL PRBs per UE` cap (applies live to all cells).
  - Slice share sliders (fractions 0–1 per slice). Sum > 1 is normalized; < 1 leaves some PRBs unused.
  - “Move RB” buttons to shift PRBs between slices (paper-like actions). Default step moves 3 PRBs per click; change in `network_layer/xApps/xapp_live_kpi_dashboard.py` by editing `SLICE_MOVE_STEP_PRBS`.
  - Optional history range slider, plot history window size, and CSV logging (see below).

Slice share semantics:

- For each cell, `quota[slice] = floor(max_dl_prb_cell × share[slice])`.
- UEs in a slice share only that slice’s quota. Baseline 1 PRB/UE if possible, remainder proportional to demand.
- Per‑UE cap is enforced after slice allocation.

---

### KPI History, Range Slider, and Logging

Enable interactive history navigation on charts (range slider), configure how many points the live plots keep in memory, and optionally persist KPIs to CSV.

- CLI flags:
  - `--kpi-history`: enable a per‑chart range slider and preserve zoom/pan across live updates.
  - `--kpi-max-points <N>`: number of points kept in memory for plots (default 50). Use `0` for unbounded history.
  - `--kpi-log`: write per‑step UE/Cell KPIs to CSV files.
  - `--kpi-log-dir <path>`: output directory for KPI CSVs (default `backend/kpi_logs`).

- Environment variables (alternative):
  - `RAN_KPI_HISTORY_ENABLE=1`
  - `RAN_KPI_MAX_POINTS=<N>` (0 = unbounded)
  - `RAN_KPI_LOG_ENABLE=1`
  - `RAN_KPI_LOG_DIR=<path>`

- Example (server mode):
```bash
python backend/main.py --preset simple --mode server \
  --kpi-history --kpi-max-points 10000 --kpi-log --kpi-log-dir backend/kpi_logs
# KPI dashboard at http://localhost:8061
```

Notes:
- With the history slider enabled, legends are placed at the top to avoid overlap with the slider.
- Unbounded history (0) grows with runtime and number of UEs; prefer a large but finite window for long runs (e.g., 5000–20000).
- CSV logs include one row per UE and per cell per step. UE CSV columns: `sim_step, imsi, dl_bps, dl_mbps, sinr_db, cqi, dl_buffer_bytes, dl_prb_granted, dl_prb_requested, dl_latency_ms`. Cell CSV: `sim_step, cell_id, dl_load, allocated_prb, max_prb`.

---

## 🧠 Example xApps

Example xApps are located in the `network_layer/xApps/` directory:

- Blind Handover xApp: Implements handover decisions based on RRC Event A3.
- AI service monitoring xApp: Monitors the AI service performance and provides insights.
 - Live KPI Dashboard xApp: Real‑time UE/Cell KPIs with per‑UE cap and per‑slice PRB controls.
 - DQN PRB Allocator xApp: Learns to shift DL PRBs among slices (eMBB/URLLC/mMTC) using a DQN policy inspired by the Tractor paper.

To load custom xApps, add them to the xApps/ directory and ensure they inherit from the xAppBase class.

### DQN PRB Allocator xApp

The DQN xApp implements a small DQN agent that observes per‑cell state and applies the 7 actions from Table 3 in the Tractor paper (move one PRB between slices or keep).

- State per cell: `[#mMTC UEs, #URLLC UEs, #eMBB UEs, PRBs_mMTC, PRBs_URLLC]` (eMBB PRBs are implied by the total).
- Actions: `0=keep, 1=mMTC→URLLC, 2=mMTC→eMBB, 3=URLLC→mMTC, 4=URLLC→eMBB, 5=eMBB→mMTC, 6=eMBB→URLLC`.
- Reward: weighted sum of per‑slice scores (eMBB queue‑drain, URLLC queueing delay proxy, mMTC utilization/idle penalty) normalized to [0,1].

Enable it at runtime (requires `torch`):

```bash
pip install torch  # if not already installed

# Server mode with DQN (online training), the KPI dashboard, and range slider
python backend/main.py --preset simple --mode server \
  --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1 \
  --kpi-history --kpi-log
```

Useful flags/env vars:
- `--dqn-prb` (or `DQN_PRB_ENABLE=1`): enable the DQN xApp.
- `--dqn-train` (or `DQN_PRB_TRAIN=1`): online training; otherwise runs greedy inference.
- `--dqn-model <path>` (or `DQN_PRB_MODEL_PATH`): save/load model weights (default `backend/models/dqn_prb.pt`).
- `--dqn-period <steps>` (`DQN_PRB_DECISION_PERIOD_STEPS`): act every N sim steps (default 1).
- `--dqn-move-step <PRBs>` (`DQN_PRB_MOVE_STEP`): PRBs moved per action (default 1).
- Exploration and learning hyper‑params can be set via env vars: `DQN_PRB_EPSILON_START`, `DQN_PRB_EPSILON_END`, `DQN_PRB_EPSILON_DECAY`, `DQN_PRB_LR`, `DQN_PRB_BATCH`, `DQN_PRB_BUFFER`, `DQN_PRB_GAMMA`.

Notes:
- The xApp applies actions after each simulator step; changes take effect in the next allocation round.
- Rewards use current KPIs and are a practical instantiation of the paper’s formulas; you can refine the shaping or weights in `settings/ran_config.py`.
- If `torch` is unavailable, the xApp disables itself gracefully.

#### DQN Training Telemetry (TensorBoard / W&B)

You can visualize training with TensorBoard or Weights & Biases (optional):

- TensorBoard (recommended locally):
  - Install: `pip install tensorboard`
  - Run with logging:
    ```bash
    python backend/main.py --preset simple --mode server \
      --dqn-prb --dqn-train --dqn-log-tb --dqn-tb-dir backend/tb_logs \
      --kpi-history
    ```
  - Launch TensorBoard: `tensorboard --logdir backend/tb_logs`
  - You’ll see per‑cell reward, slice scores (eMBB/URLLC/mMTC), loss, epsilon, PRB quotas, and action histograms.

- Weights & Biases (cloud):
  - Install and login: `pip install wandb && wandb login`
  - Run with logging:
    ```bash
    python backend/main.py --preset simple --mode server \
      --dqn-prb --dqn-train --dqn-wandb \
      --dqn-wandb-project ai-ran-dqn --dqn-wandb-name local-run-1
    ```
  - The same metrics are logged to your W&B project.

- Sample code run in tensorboard:
```
    python backend/main.py --preset simple --mode server   --ue-embb 10 --ue-urllc 10 --ue-mmtc 10 --freeze-mobility   --trace-raw-map slice:eMBB:backend/assets/traces/eMBB_aligned.csv:172.30.1.1   --trace-raw-map slice:URLLC:backend/assets/traces/URLLC_aligned.csv:172.30.1.1   --trace-raw-map slice:mMTC:backend/assets/traces/mMTC_aligned.csv:172.30.1.1   --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic --trace-loop   --dqn-prb --dqn-train --dqn-period 1 --dqn-move-step 1   --kpi-history --kpi-log --dqn-log-tb --dqn-tb-dir backend/tb_logs
```
---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests to improve the simulator.

---

## 📬 Contact

For questions or support, please feel free to open issues.
