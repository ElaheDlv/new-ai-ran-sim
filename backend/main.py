import os
import argparse
import asyncio
import json
from functools import partial

import dotenv

# -------------------------------------------------------------
# CLI args to control topology preset and run mode before import
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="AI-RAN Simulator Backend")
parser.add_argument("--preset", choices=["default", "simple"], help="Topology preset")
parser.add_argument("--ue-max", type=int, help="Override UE_DEFAULT_MAX_COUNT")
parser.add_argument("--ue-embb", type=int, help="Simple preset: number of UEs with eMBB subscription")
parser.add_argument("--ue-urllc", type=int, help="Simple preset: number of UEs with URLLC subscription")
parser.add_argument("--ue-mmtc", type=int, help="Simple preset: number of UEs with mMTC subscription")
parser.add_argument(
    "--subscribe",
    "-S",
    action="append",
    help=(
        "Create AI service subscriptions at startup. Repeatable. "
        "Format: service_name:IMSI_0,IMSI_1 (no spaces)."
    ),
)
parser.add_argument(
    "--subscribe-file",
    help=(
        "Path to JSON file with a list of {\"service\": name, \"ues\": [..]} objects."
    ),
)
parser.add_argument(
    "--ensure-ues",
    action="store_true",
    help=(
        "If set, auto-register any UE IDs mentioned in --subscribe that are not present yet."
    ),
)
parser.add_argument(
    "--mode", choices=["server", "headless"], default="server", help="Run as WebSocket server or headless loop",
)
parser.add_argument("--steps", type=int, default=120, help="Headless: number of steps to run")
parser.add_argument(
    "--trace-map",
    action="append",
    help=(
        "Attach CSV traffic traces to UEs. Repeatable. Format: IMSI_#:path.csv (t_s,dl_bytes[,ul_bytes])."
    ),
)
parser.add_argument(
    "--trace-json",
    help=(
        "JSON file with a list of {\"imsi\": \"IMSI_#\", \"file\": \"path.csv\", \"speedup\": 1.0}."
    ),
)
parser.add_argument(
    "--trace-speedup",
    type=float,
    default=1.0,
    help="Global speedup factor for traces (optional; per-item overrides).",
)

parser.add_argument(
    "--strict-real-traffic",
    action="store_true",
    help="If set, only real offered traffic (traces/AI-service) is shown; no fallback achievable rate.",
)


parser.add_argument(
    "--trace-raw-map",
    action="append",
    help=(
        "Attach RAW packet CSV traces and aggregate on the fly. Repeatable. "
        "Format: IMSI_#:path.csv:UE_IP"
    ),
)
parser.add_argument(
    "--trace-bin",
    type=float,
    default=1.0,
    help="Bin size for raw packet CSV aggregation in seconds (default 1.0)",
)
parser.add_argument(
    "--trace-slice-dir",
    help=(
        "Directory containing per-slice traces; auto-picks files matching embb*.csv, urllc*.csv, mmtc*.csv. "
        "If not given, you can specify explicit files via --trace-slice-embb/--trace-slice-urllc/--trace-slice-mmtc."
    ),
)
parser.add_argument("--trace-slice-embb", help="CSV to attach to ALL eMBB UEs")
parser.add_argument("--trace-slice-urllc", help="CSV to attach to ALL URLLC UEs")
parser.add_argument("--trace-slice-mmtc", help="CSV to attach to ALL mMTC UEs")
parser.add_argument(
    "--trace-slice-ueip",
    help=(
        "UE IP used in raw packet CSVs for DL/UL classification (e.g., 172.30.1.1). "
        "If omitted, defaults to 172.30.1.1."
    ),
)
args, unknown = parser.parse_known_args()

if args.preset:
    os.environ["RAN_TOPOLOGY_PRESET"] = args.preset

# If per-slice counts are provided in simple preset, align total UEs accordingly
slice_counts = [c for c in (args.ue_embb, args.ue_urllc, args.ue_mmtc) if c is not None]
if slice_counts and (args.preset == "simple"):
    total_from_slices = sum(slice_counts)
    if args.ue_max is not None and args.ue_max != total_from_slices:
        print(
            f"[main] --ue-max ({args.ue_max}) != sum of slice counts ({total_from_slices}); adjusting total to {total_from_slices}."
        )
    os.environ["UE_DEFAULT_MAX_COUNT"] = str(total_from_slices)
elif args.ue_max is not None:
    os.environ["UE_DEFAULT_MAX_COUNT"] = str(args.ue_max)

if args.ue_embb is not None:
    os.environ["UE_SIMPLE_COUNT_EMBB"] = str(args.ue_embb)
if args.ue_urllc is not None:
    os.environ["UE_SIMPLE_COUNT_URLLC"] = str(args.ue_urllc)
if args.ue_mmtc is not None:
    os.environ["UE_SIMPLE_COUNT_MMTC"] = str(args.ue_mmtc)
if args.strict_real_traffic:
    os.environ["SIM_TRAFFIC_STRICT_REAL_ONLY"] = "1"

# -------------------------------------------------------------
# Parse subscription specifications from CLI / file (deferred use)
# -------------------------------------------------------------
SUBSCRIBE_SPECS = []  # list of {"service": name, "ues": ["IMSI_0", ...]}
def _parse_subscribe_specs():
    specs = []
    if args.subscribe:
        for s in args.subscribe:
            try:
                name, ue_csv = s.split(":", 1)
                ues = [u.strip() for u in ue_csv.split(",") if u.strip()]
                if name and ues:
                    specs.append({"service": name.strip(), "ues": ues})
            except ValueError:
                print(f"[main] Ignoring invalid --subscribe spec: {s}")
    if args.subscribe_file:
        try:
            import json
            with open(args.subscribe_file, "r") as f:
                data = json.load(f)
            for item in data:
                name = item.get("service")
                ues = item.get("ues", [])
                if name and isinstance(ues, list) and ues:
                    specs.append({"service": name, "ues": ues})
        except Exception as e:
            print(f"[main] Failed to read --subscribe-file: {e}")
    return specs

SUBSCRIBE_SPECS = _parse_subscribe_specs()

# -------------------------------------------------------------
# Parse trace attachments
# -------------------------------------------------------------
TRACE_SPECS = []  # list of {"imsi": str, "file": str, "speedup": float}
def _parse_trace_specs():
    specs = []
    if args.trace_map:
        for s in args.trace_map:
            try:
                imsi, path = s.split(":", 1)
                imsi = imsi.strip(); path = path.strip()
                if imsi and path:
                    specs.append({"imsi": imsi, "file": path, "speedup": args.trace_speedup})
            except ValueError:
                print(f"[main] Ignoring invalid --trace-map spec: {s}")
    if args.trace_json:
        try:
            import json
            with open(args.trace_json, "r") as f:
                data = json.load(f)
            for item in data:
                imsi = item.get("imsi"); path = item.get("file"); sp = float(item.get("speedup", args.trace_speedup))
                if imsi and path:
                    specs.append({"imsi": imsi, "file": path, "speedup": sp})
        except Exception as e:
            print(f"[main] Failed to read --trace-json: {e}")
    return specs

TRACE_SPECS = _parse_trace_specs()

# Raw packet trace specs
RAW_TRACE_SPECS = []  # list of {"imsi": str, "file": str, "ue_ip": str, "bin_s": float, "speedup": float}

def _parse_raw_trace_specs():
    specs = []
    if args.trace_raw_map:
        for s in args.trace_raw_map:
            try:
                imsi, path, ueip = s.split(":", 2)
                imsi = imsi.strip(); path = path.strip(); ueip = ueip.strip()
                if imsi and path and ueip:
                    specs.append({
                        "imsi": imsi,
                        "file": path,
                        "ue_ip": ueip,
                        "bin_s": args.trace_bin,
                        "speedup": args.trace_speedup,
                    })
            except ValueError:
                print(f"[main] Ignoring invalid --trace-raw-map spec: {s}")
    return specs

RAW_TRACE_SPECS = _parse_raw_trace_specs()

# Slice-level traces used for every UE of a given slice
SLICE_TRACE_SPECS = {}

def _parse_slice_trace_specs():
    import os, glob
    specs = {}
    # explicit files have priority
    if args.trace_slice_embb:
        specs["eMBB"] = args.trace_slice_embb
    if args.trace_slice_urllc:
        specs["URLLC"] = args.trace_slice_urllc
    if args.trace_slice_mmtc:
        specs["mMTC"] = args.trace_slice_mmtc
    # directory auto-detect (fallback for any missing slice)
    dirpath = args.trace_slice_dir
    # If no explicit dir but the standard path exists, use it as a convenience
    default_dir = os.path.join(os.path.dirname(__file__), "assets", "traces")
    if not dirpath and os.path.isdir(default_dir):
        dirpath = default_dir
    if dirpath and os.path.isdir(dirpath):
        def pick(globpat):
            hits = sorted(glob.glob(os.path.join(dirpath, globpat)))
            return hits[0] if hits else None
        specs.setdefault("eMBB", pick("embb*.csv"))
        specs.setdefault("URLLC", pick("urllc*.csv"))
        specs.setdefault("mMTC", pick("mmtc*.csv"))
    # Drop Nones
    specs = {k: v for k, v in specs.items() if v}
    return specs

SLICE_TRACE_SPECS = _parse_slice_trace_specs()


# Load .env (won't override already-set env)
dotenv.load_dotenv()

"""
Note on optional deps:
- Server mode requires `websockets`.
- xApps (like the KPI dashboard) require `dash` and friends as declared in environment.yml.
We import `websockets` lazily to avoid blocking headless runs when it isn't installed.
"""
try:
    import websockets as _websockets
except Exception:
    _websockets = None

import settings
from utils import (
    setup_logging,
)
from network_layer.simulation_engine import SimulationEngine
from knowledge_layer import KnowledgeRouter
# Server-mode imports (agents, streaming) are done lazily in websocket_handler

setup_logging()


# Server-only: command handlers are constructed lazily inside websocket_handler.


async def _bootstrap_subscriptions_if_any(simulation_engine, knowledge_router):
    """Create CLI-defined AI service subscriptions and optionally ensure UEs exist."""
    if not SUBSCRIBE_SPECS:
        return
    # Optionally register missing UEs referenced by subscriptions
    if args.ensure_ues:
        from settings.slice_config import NETWORK_SLICE_EMBB_NAME
        for spec in SUBSCRIBE_SPECS:
            for ue_id in spec["ues"]:
                if ue_id not in simulation_engine.ue_list:
                    subs = settings.CORE_UE_SUBSCRIPTION_DATA.get(ue_id, [NETWORK_SLICE_EMBB_NAME])
                    simulation_engine.register_ue(
                        ue_imsi=ue_id,
                        subscribed_slices=subs,
                        register_slice=subs[0],
                    )
    # Create subscriptions
    for spec in SUBSCRIBE_SPECS:
        name = spec["service"]
        ues = spec["ues"]
        ai_data = knowledge_router.query_knowledge(f"/ai_services/{name}/raw")
        if not ai_data:
            print(f"[main] Unknown AI service '{name}'. Skipping subscription.")
            continue
        simulation_engine.ric.ai_service_subscription_manager.create_subscription(
            ai_service_name=name,
            ai_service_data=ai_data,
            ue_id_list=ues,
        )


def _print_available_ai_services(knowledge_router):
    try:
        overview = knowledge_router.query_knowledge("/ai_services")
        if isinstance(overview, str):
            print("\n===== Available AI Services =====\n")
            print(overview)
            print("\n================================\n")
    except Exception as e:
        print(f"[main] Failed to list AI services: {e}")




def _attach_traces_if_any(simulation_engine):
    if not TRACE_SPECS and not RAW_TRACE_SPECS:
        # Still allow slice-level traces
        if not SLICE_TRACE_SPECS:
            return
    from utils import load_csv_trace, load_raw_packet_csv
    from settings.slice_config import NETWORK_SLICE_EMBB_NAME
    # RAW packet traces first (aggregate on the fly)
    for spec in RAW_TRACE_SPECS:
        imsi = spec["imsi"]; path = spec["file"]; speed = float(spec.get("speedup", 1.0))
        ue = simulation_engine.ue_list.get(imsi)
        if ue is None:
            subs = settings.CORE_UE_SUBSCRIPTION_DATA.get(imsi, [NETWORK_SLICE_EMBB_NAME])
            simulation_engine.register_ue(imsi, subs, register_slice=subs[0])
            ue = simulation_engine.ue_list.get(imsi)
        if ue is None:
            print(f"[main] Could not attach RAW trace to {imsi}: UE not present")
            continue
        try:
            samples = load_raw_packet_csv(path, ue_ip=spec["ue_ip"], bin_s=float(spec.get("bin_s", 1.0)))
            ue.attach_trace(samples, speed)
            print(f"[main] Attached RAW trace {path} (n={len(samples)}) to {imsi}, speedup={speed}")
        except Exception as e:
            print(f"[main] Failed to load RAW trace for {imsi} from {path}: {e}")
    # Pre-aggregated traces
    for spec in TRACE_SPECS:
        imsi = spec["imsi"]; path = spec["file"]; speed = float(spec.get("speedup", 1.0))
        ue = simulation_engine.ue_list.get(imsi)
        if ue is None:
            subs = settings.CORE_UE_SUBSCRIPTION_DATA.get(imsi, [NETWORK_SLICE_EMBB_NAME])
            simulation_engine.register_ue(imsi, subs, register_slice=subs[0])
            ue = simulation_engine.ue_list.get(imsi)
        if ue is None:
            print(f"[main] Could not attach trace to {imsi}: UE not present")
            continue
        try:
            samples = load_csv_trace(path)
            ue.attach_trace(samples, speed)
            print(f"[main] Attached trace {path} (n={len(samples)}) to {imsi}, speedup={speed}")
        except Exception as e:
            print(f"[main] Failed to load trace for {imsi} from {path}: {e}")
    # Slice-level traces: apply to all existing and future UEs per slice
    if SLICE_TRACE_SPECS:
        try:
            mapping = {}
            for slice_name, path in SLICE_TRACE_SPECS.items():
                try:
                    # Heuristic: try raw-packet loader first (most common for these files),
                    # fall back to pre-aggregated loader if that fails.
                    ueip = args.trace_slice_ueip
                    if ueip is None:
                        # Quick auto-detect: pick the most common Source IP in the CSV
                        try:
                            import csv
                            from collections import Counter
                            cnt = Counter()
                            with open(path, 'r') as f:
                                reader = csv.DictReader(f)
                                for i, row in enumerate(reader):
                                    s = row.get('Source') or row.get('source')
                                    if s:
                                        cnt[s] += 1
                                    if i > 20000:
                                        break
                            if cnt:
                                ueip = cnt.most_common(1)[0][0]
                        except Exception:
                            ueip = None
                    ueip = ueip or "172.30.1.1"
                    try:
                        samples = load_raw_packet_csv(path, ue_ip=ueip, bin_s=float(args.trace_bin))
                    except Exception:
                        samples = load_csv_trace(path)
                    mapping[slice_name] = (samples, float(args.trace_speedup))
                    print(f"[main] Loaded slice-trace for {slice_name}: {path} (n={len(samples)})")
                except Exception as e:
                    print(f"[main] Failed to load slice-trace for {slice_name} from {path}: {e}")
            if mapping:
                simulation_engine.configure_slice_traces(mapping)
        except Exception as e:
            print(f"[main] Could not configure slice traces: {e}")
async def websocket_handler(websocket):
    # Import server-only helpers lazily to avoid unnecessary hard deps in headless mode
    from utils.websocket_utils import (
        WebSocketResponse,
        handle_start_simulation,
        handle_stop_simulation,
        handle_get_simulation_state,
        handle_get_routes,
        handle_query_knowledge,
        stream_agent_chat,
        handle_network_user_action,
        WebSocketSingleton,
    )
    WebSocketSingleton().set_websocket(websocket)
    simulation_engine = SimulationEngine()
    simulation_engine.reset_network()
    simulation_engine.network_setup()
    knowledge_router = KnowledgeRouter()
    knowledge_router.import_routes(simulation_engine)
    _print_available_ai_services(knowledge_router)
    # Attach traces in server mode (after network init)
    _attach_traces_if_any(simulation_engine)
    # Bootstrap CLI-defined subscriptions (server mode)
    await _bootstrap_subscriptions_if_any(simulation_engine, knowledge_router)
    # Build command handlers for this server instance
    # Import server-only agent handlers lazily
    from intelligence_layer import engineer_chat_agent
    from intelligence_layer.ai_service_pipeline import handle_ai_service_pipeline_chat

    COMMAND_HANDLERS = {
        ("network_layer", "start_simulation"): handle_start_simulation,
        ("network_layer", "stop_simulation"): handle_stop_simulation,
        ("network_layer", "get_simulation_state"): handle_get_simulation_state,
        ("knowledge_layer", "get_routes"): handle_get_routes,
        ("knowledge_layer", "query_knowledge"): handle_query_knowledge,
        ("intelligence_layer", "ai_service_pipeline"): handle_ai_service_pipeline_chat,
        ("intelligence_layer", "network_engineer_chat"): partial(
            stream_agent_chat,
            command="network_engineer_chat_response",
            agent_func=engineer_chat_agent,
        ),
        ("intelligence_layer", "network_user_action"): handle_network_user_action,
    }

    while True:
        message = await websocket.recv()
        try:
            message_json = json.loads(message)
            layer = message_json.get("layer")
            command = message_json.get("command")
            data = message_json.get("data", {})
        except (json.JSONDecodeError, KeyError):
            response = WebSocketResponse(
                layer=None, command=None, response=None, error="Invalid message format"
            )
            await websocket.send(response.to_json())
            continue

        try:
            handler = COMMAND_HANDLERS.get((layer, command))
            if handler:
                await handler(
                    websocket=websocket,
                    simulation_engine=simulation_engine,
                    knowledge_router=knowledge_router,
                    data=data,
                )
            else:
                response = WebSocketResponse(
                    layer=layer,
                    command=command,
                    response=None,
                    error=f"Unknown command: {command}",
                )
                await websocket.send(response.to_json())
        except Exception as e:
            response = WebSocketResponse(
                layer=layer, command=command, response=None, error=str(e)
            )
            await websocket.send(response.to_json())


async def main():
    # Headless mode: run simulation loop without WebSocket server
    if args.mode == "headless":
        eng = SimulationEngine()
        eng.reset_network()
        eng.network_setup()
        # Bootstrap CLI-defined subscriptions (headless mode)
        kr = KnowledgeRouter(); kr.import_routes(eng)
        _print_available_ai_services(kr)
        await _bootstrap_subscriptions_if_any(eng, kr)
        _attach_traces_if_any(eng)
        for _ in range(args.steps):
            eng.sim_step += 1
            eng.step(settings.SIM_STEP_TIME_DEFAULT)
            await asyncio.sleep(settings.SIM_STEP_TIME_DEFAULT)
        print(
            f"Headless run completed. BS: {list(eng.base_station_list.keys())}, Cells: {list(eng.cell_list.keys())}"
        )
        return

    # Server mode (default)
    if _websockets is None:
        print(
            "[main] Missing dependency: websockets. Install it to use server mode: pip install websockets"
        )
        return
    async with _websockets.serve(
        websocket_handler, settings.WS_SERVER_HOST, settings.WS_SERVER_PORT
    ):
        print(
            f"WebSocket server started on ws://{settings.WS_SERVER_HOST}:{settings.WS_SERVER_PORT}"
        )
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
