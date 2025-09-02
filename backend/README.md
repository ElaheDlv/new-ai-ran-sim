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

You can explicitly choose how many UEs are subscribed to each slice when using the simple preset. These UEs will attach to their single subscribed slice deterministically.

- With CLI flags:

```bash
python main.py --preset simple --ue-max 10 \
  --ue-embb 6 --ue-urllc 3 --ue-mmtc 1 --mode headless --steps 120
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
- Counts are capped so they never exceed `UE_DEFAULT_MAX_COUNT` and any remainder fills eMBB.
- Runtime spawn is still dynamic (1–2 per step in simple mode); slice membership is fixed per IMSI.

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
