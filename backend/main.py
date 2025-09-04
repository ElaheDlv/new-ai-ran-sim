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
    "--freeze-mobility",
    action="store_true",
    help="Freeze UE mobility (speed=0; pin targets to current position).",
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
if args.freeze_mobility:
    os.environ["SIM_FREEZE_MOBILITY"] = "1"

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


async def websocket_handler(websocket):
    WebSocketSingleton().set_websocket(websocket)
    simulation_engine = SimulationEngine()
    simulation_engine.reset_network()
    simulation_engine.network_setup()
    knowledge_router = KnowledgeRouter()
    knowledge_router.import_routes(simulation_engine)
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
