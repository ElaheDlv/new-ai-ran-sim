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

# Load .env (won't override already-set env)
dotenv.load_dotenv()

import websockets
import settings
from utils import (
    WebSocketResponse,
    handle_start_simulation,
    handle_stop_simulation,
    handle_get_simulation_state,
    handle_get_routes,
    handle_query_knowledge,
    stream_agent_chat,
    handle_network_user_action,
    setup_logging,
    WebSocketSingleton,
)
from network_layer.simulation_engine import SimulationEngine
from knowledge_layer import KnowledgeRouter
from intelligence_layer import engineer_chat_agent
from intelligence_layer.ai_service_pipeline import handle_ai_service_pipeline_chat

setup_logging()


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


async def websocket_handler(websocket):
    WebSocketSingleton().set_websocket(websocket)
    simulation_engine = SimulationEngine()
    simulation_engine.reset_network()
    simulation_engine.network_setup()
    knowledge_router = KnowledgeRouter()
    knowledge_router.import_routes(simulation_engine)
    _print_available_ai_services(knowledge_router)
    # Bootstrap CLI-defined subscriptions (server mode)
    await _bootstrap_subscriptions_if_any(simulation_engine, knowledge_router)
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
        for _ in range(args.steps):
            eng.sim_step += 1
            eng.step(settings.SIM_STEP_TIME_DEFAULT)
            await asyncio.sleep(settings.SIM_STEP_TIME_DEFAULT)
        print(
            f"Headless run completed. BS: {list(eng.base_station_list.keys())}, Cells: {list(eng.cell_list.keys())}"
        )
        return

    # Server mode (default)
    async with websockets.serve(
        websocket_handler, settings.WS_SERVER_HOST, settings.WS_SERVER_PORT
    ):
        print(
            f"WebSocket server started on ws://{settings.WS_SERVER_HOST}:{settings.WS_SERVER_PORT}"
        )
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
