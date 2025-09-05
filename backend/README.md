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
- Dependencies (recommended via Conda):

```bash
conda env create -f environment.yml
conda activate airansim
```

Notes:
- Headless mode can run without optional extras; if `dash` is not installed the KPI xApp disables itself. Server mode requires `websockets` (the app prints a helpful message if missing).

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


### Generate real AI‑service traffic (no frontend)

You can bootstrap AI service subscriptions from the CLI. Subscribed UEs will periodically send HTTP requests to the deployed edge container, and logs will include process times and endpoints.

- One‑liner (headless):

```bash
python main.py --preset simple --mode headless --steps 300 \
  --subscribe ultralytics-yolov8-yolov8s:IMSI_0,IMSI_1 \
  --subscribe trpakov-vit-face-expression:IMSI_2 \
  --ensure-ues
```

- Flags:
  - `--subscribe name:IMSI_A,IMSI_B` (repeatable): create a subscription for service `name` and the listed UEs.
  - `--subscribe-file path.json`: JSON list of `{"service": name, "ues": [..]}` objects.
  - `--ensure-ues`: auto‑register any listed UE IDs that don’t exist yet.

Notes:
- In server mode the subscriptions are created after the network initializes; start the simulation from your client as usual.
- The KPI xApp is independent; you can still open http://localhost:8061 to watch KPIs.


### Use the frontend and CLI together

Run the backend in server mode so the WebSocket API is available to the UI, while also pre‑creating AI‑service subscriptions via CLI.

```bash
python main.py --preset simple --mode server \
  --subscribe ultralytics-yolov8-yolov8s:IMSI_0,IMSI_1 \
  --ensure-ues

# In another terminal
cd frontend
npm run dev
```

CLI and UI subscriptions both go to the same manager; duplicates are ignored safely.

### Combine slice mix and subscriptions

You can set the UE slice mix and also subscribe specific UEs to services in the same run.

Server (UI + CLI):

```bash
python main.py --preset simple --ue-embb 6 --ue-urllc 3 --ue-mmtc 1 --mode server \
  --subscribe ultralytics-yolov8-yolov8s:IMSI_0,IMSI_1 \
  --ensure-ues
```

Headless:

```bash
python main.py --preset simple --ue-embb 6 --ue-urllc 3 --ue-mmtc 1 --mode headless --steps 300 \
  --subscribe ultralytics-yolov8-yolov8s:IMSI_0,IMSI_3 \
  --subscribe trpakov-vit-face-expression:IMSI_2 \
  --ensure-ues
```

Additional notes:
- Total UEs equals the sum of `--ue-embb/--ue-urllc/--ue-mmtc` in simple preset (if `--ue-max` differs, it is aligned to the sum).
- IMSI→slice assignment (simple preset): first `mMTC` as `IMSI_0..`, then `URLLC`, then remaining as `eMBB`.
- `--ensure-ues` registers any listed IMSIs that don’t exist yet so they immediately generate traffic.
- The backend prints the available AI services at startup; prefer those names to avoid Docker pull failures.


### Quick: Use CSV Traces (real offered traffic)

Drive per‑UE offered load from CSV files. Two options:

- Pre‑aggregated CSV (columns: `t_s,dl_bytes[,ul_bytes]`):

  Headless
  ```bash
  python main.py --preset simple --mode headless --steps 180 \
    --trace-map IMSI_0:backend/assets/traces/embb_youtube.csv \
    --trace-map IMSI_1:backend/assets/traces/urllc_meet.csv \
    --trace-speedup 1.0 --strict-real-traffic
  ```

  Server (UI + KPI dashboard)
  ```bash
  python main.py --preset simple --mode server \
    --trace-map IMSI_0:backend/assets/traces/embb_youtube.csv \
    --trace-map IMSI_1:backend/assets/traces/urllc_meet.csv \
    --trace-speedup 1.0 --strict-real-traffic
  ```

- Raw packet CSV (columns like `Time,Source,Destination,Length`): auto‑aggregate with UE IP

  ```bash
  python main.py --preset simple --mode headless --steps 180 \
    --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1 \
    --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1 \
    --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic
  ```

Notes:
- Place CSVs anywhere (e.g. `backend/assets/traces/`).
- `--strict-real-traffic` shows only real served traffic; without it, UEs without traces display fallback achievable rate.
- You can also provide a JSON map via `--trace-json`. See detailed section below.

For full details (CSV format, JSON mapping, dashboard tips), see “📈 Replay Real App Traces (YouTube/Meet/mMTC)” below.



---

## 📊 Live KPI Dashboard xApp

Drop-in xApp that starts a Dash server at `http://localhost:8061` and streams per‑UE and per‑cell KPIs.

- Per‑UE: bitrate (Mbps), SINR, CQI, allocated PRBs.
- Per‑cell: load, PRB usage; fixed PRB quotas per slice (eMBB/URLLC/mMTC).
- Controls:
  - `Max DL PRBs per UE` cap (applies live to all cells).
  - Slice share sliders (fractions 0–1 per slice). Sum > 1 is normalized; < 1 leaves some PRBs unused.
  - “Move RB” buttons to shift PRBs between slices (paper-like actions). Default step moves 3 PRBs per click; change in `network_layer/xApps/xapp_live_kpi_dashboard.py` by editing `SLICE_MOVE_STEP_PRBS`.

Slice share semantics:

- For each cell, `quota[slice] = floor(max_dl_prb_cell × share[slice])`.
- UEs in a slice share only that slice’s quota. Baseline 1 PRB/UE if possible, remainder proportional to demand.
- Per‑UE cap is enforced after slice allocation.

---

## 📈 Replay Real App Traces (YouTube/Meet/mMTC)

You can drive realistic per‑UE offered load by attaching CSV traces. Each CSV should have:

- Required columns: `t_s, dl_bytes`
- Optional: `ul_bytes`
- `t_s` is seconds (any origin; the loader normalizes to start at 0)

Example rows:

```
t_s,dl_bytes,ul_bytes
0,1450000,12000
1,1210000,11000
2,1980000,15000
```

Attach pre‑aggregated traces at startup:

```bash
python main.py --preset simple --mode headless --steps 180   --trace-map IMSI_0:/data/embb_youtube.csv   --trace-map IMSI_1:/data/urllc_meet.csv   --trace-speedup 1.0
```

Or via JSON (pre‑aggregated):

```json
[
  {"imsi": "IMSI_0", "file": "/data/embb_youtube.csv", "speedup": 1.0},
  {"imsi": "IMSI_1", "file": "/data/urllc_meet.csv"}
]
```

Run with:

```bash
python main.py --preset simple --mode headless --steps 180   --trace-json trace_map.json
```

How it works:

- Each UE enqueues `dl_bytes` from its trace into a per‑UE buffer every step.
- Cells serve the buffer according to CQI→MCS→PRBs; actual served bitrate is limited by both capacity and buffer.
- The dashboard’s “DL buffer (bytes)” lets you observe backlog vs service.

Sanity checks:

- Offered_total ≈ Served_total + DL_buffer (non‑negative, within rounding).
- With high capacity, buffer ~0 and served ≈ offered; with tight capacity, buffer grows roughly at (offered − capacity).


Where to place files:

- Put traces under `backend/assets/traces/` (any path is fine; this is a convenient default).
- Each CSV you pass via `--trace-map` or `--trace-raw-map` is attached to one UE (one file → one IMSI).
- You can reuse the same file for multiple UEs by repeating the flag.

Combine with frontend and AI services:

- Run the backend in server mode with trace flags; start the frontend normally. The KPI dashboard continues to work.
- You can also preload edge AI‑service subscriptions in the same run, e.g.:

```bash
python main.py --preset simple --mode server   --trace-raw-map IMSI_0:backend/assets/traces/embb_04_10.csv:172.30.1.1   --subscribe ultralytics-yolov8-yolov8s:IMSI_1   --ensure-ues
```

CLI flag quick reference (traces):

- `--trace-map IMSI_#:file.csv`               Pre‑aggregated CSV (t_s, dl_bytes[, ul_bytes])
- `--trace-json path.json`                    JSON list of {imsi, file, speedup}
- `--trace-raw-map IMSI_#:raw.csv:UE_IP`      Raw packet CSV; auto‑aggregate by `--trace-bin`
- `--trace-bin <seconds>`                     Bin size for raw CSV aggregation (default 1.0)
- `--trace-speedup <x>`                       Time scaling for trace playback (default 1.0)
- `--strict-real-traffic`                     Only show real served traffic (no fallback achievable rate)
- `--trace-slice-dir <path>`                  Auto‑attach per‑slice traces found in a directory (embb*.csv, urllc*.csv, mmtc*.csv)
- `--trace-slice-embb/--trace-slice-urllc/--trace-slice-mmtc`  Explicit files for each slice (applies to ALL UEs of that slice)
- `--trace-slice-ueip <IP>`                   UE IP for raw packet CSVs (used to separate DL/UL); auto‑detected if omitted


---

## 🎯 Minimal 3‑UE Per‑Slice Trace Demo

Spin up exactly three UEs (one per slice) and replay a trace for each to verify realistic traffic end‑to‑end.

Slice → IMSI mapping in simple preset with `--ue-embb 1 --ue-urllc 1 --ue-mmtc 1`:

- `IMSI_0` → mMTC
- `IMSI_1` → URLLC
- `IMSI_2` → eMBB

Server mode (frontend + KPI dashboard):

```bash
python main.py --preset simple --mode server   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1   --trace-raw-map IMSI_0:backend/assets/traces/mmtc_04_10.csv:172.30.1.1   --trace-bin 1.0 --trace-speedup 1.0

# In another terminal
cd frontend
npm run dev
```

Headless (no frontend):

```bash
python main.py --preset simple --mode headless --steps 180   --ue-embb 1 --ue-urllc 1 --ue-mmtc 1   --trace-raw-map IMSI_2:backend/assets/traces/embb_04_10.csv:172.30.1.1   --trace-raw-map IMSI_1:backend/assets/traces/urllc_04_10.csv:172.30.1.1   --trace-raw-map IMSI_0:backend/assets/traces/mmtc_04_10.csv:172.30.1.1   --trace-bin 1.0 --trace-speedup 1.0
```

Tips:

- The third field in `--trace-raw-map` is the UE IP used in the capture; adjust if your CSVs use a different device IP.
- In the KPI dashboard, look at “DL buffer (bytes)” (should rise/fall with the trace) and per‑UE “DL Mbps” (served rate). For a strict “only real traffic shown” option, ask to enable the strict mode (no fallback achievable rate).

### New: Apply a directory of per‑slice traces to all UEs

If your traces are stored under `backend/assets/traces` and named like `embb_*.csv`, `urllc_*.csv`, `mmtc_*.csv`, you can auto‑attach them to every UE according to its slice (both existing and newly‑spawned UEs). For raw packet CSVs, pass the device IP used in the capture (or rely on auto‑detection).

Headless
```bash
python main.py --preset simple --mode headless --steps 180 \
  --trace-slice-dir backend/assets/traces \
  --trace-slice-ueip 172.30.1.250 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic
```

Server
```bash
python main.py --preset simple --mode server \
  --trace-slice-dir backend/assets/traces \
  --trace-slice-ueip 172.30.1.250 \
  --trace-bin 1.0 --trace-speedup 1.0 --strict-real-traffic
```

Notes:
- Omit `--trace-slice-ueip` to auto‑detect the UE IP from the CSVs (picks the most common `Source` IP).
- The mapping is applied per slice and persists for all future UE spawns. You can still mix per‑IMSI traces using `--trace-map`.

---

## 🧠 Example xApps

Example xApps are located in the `network_layer/xApps/` directory:

- Blind Handover xApp: Implements handover decisions based on RRC Event A3.
- AI service monitoring xApp: Monitors the AI service performance and provides insights.
 - Live KPI Dashboard xApp: Real‑time UE/Cell KPIs with per‑UE cap and per‑slice PRB controls.

To load custom xApps, add them to the xApps/ directory and ensure they inherit from the xAppBase class.

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests to improve the simulator.

---

## 📬 Contact

For questions or support, please feel free to open issues.
