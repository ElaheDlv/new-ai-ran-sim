import time
import numpy as np
from settings.ai_service_config import (
    get_random_ai_service_request_data,
    prepare_ai_service_sample_request,
)
from utils import (
    dist_between,
    get_rrc_measurement_event_monitor,
    dbm_to_watts,
    sinr_to_cqi,
)
from tabulate import tabulate

import settings
import logging

logger = logging.getLogger(__name__)


class UE:
    def __init__(
        self,
        ue_imsi="IMSI_1",
        operation_region={"min_x": 0, "min_y": 0, "max_x": 2000, "max_y": 2000},
        position_x=0,
        position_y=0,
        target_x=0,
        target_y=0,
        speed_mps=0,
        simulation_engine=None,
        connection_time=settings.UE_DEFAULT_TIMEOUT,
    ):
        self.ue_imsi = ue_imsi
        self.operation_region = operation_region
        self.position_x = position_x
        self.position_y = position_y
        self.target_x = target_x
        self.target_y = target_y
        self.speed_mps = speed_mps
        self.time_remaining = connection_time
        self.simulation_engine = simulation_engine

        self.slice_type = None
        self.qos_profile = None
        self.connected = False

        self.downlink_bitrate = 0
        self.downlink_latency = 0
        self.rrc_measurement_event_monitors = []
        self.downlink_received_power_dBm_dict = {}
        self.downlink_sinr = 0
        self.downlink_cqi = 0
        self.downlink_mcs_index = -1
        self.downlink_mcs_data = None

        self.uplink_bitrate = 0
        self.uplink_latency = 0
        self.uplink_transmit_power_dBm = settings.UE_TRANSMIT_POWER

        self.current_cell = None
        self.serving_cell_history = []

        self.ai_service_subscriptions = {}
        self.ai_service_request_countdonw = settings.UE_AI_SERVICE_REQUEST_COUNTDOWN
        self.ai_service_responses = {}

        # Trace-driven traffic fields (optional)
        self.dl_buffer_bytes = 0
        self.ul_buffer_bytes = 0
        self.offered_dl_bytes_total = 0
        self.served_dl_bytes_total = 0
        self._trace_samples = None  # list of (t_s, dl_bytes, ul_bytes)
        self._trace_cursor = 0
        self._trace_speedup = 1.0
        self._trace_started = False

    def __repr__(self):
        return f"UE(ue_imsi={self.ue_imsi}, \
            operation_region={self.operation_region}, \
            position=({self.position_x}, {self.position_y}), \
            current_cell={self.current_cell.cell_id if self.current_cell else None})"

    @property
    def dist_to_target(self):
        return dist_between(
            self.position_x,
            self.position_y,
            self.target_x,
            self.target_y,
        )

    @property
    def current_bs(self):
        if self.current_cell is None:
            return None
        return self.current_cell.base_station

    @property
    def target_reached(self):
        return self.dist_to_target == 0

    def set_target(self, target_x, target_y):
        assert (
            target_x >= self.operation_region["min_x"]
            and target_x <= self.operation_region["max_x"]
        ), f"Target X {target_x} is out of operation region bounds."
        assert (
            target_y >= self.operation_region["min_y"]
            and target_y <= self.operation_region["max_y"]
        ), f"Target Y {target_y} is out of operation region bounds."
        self.target_x = target_x
        self.target_y = target_y
        logger.info(
            f"UE {self.ue_imsi}: Target set to ({self.target_x}, {self.target_y})"
        )

    def set_downlink_bitrate(self, downlink_bitrate):
        self.downlink_bitrate = downlink_bitrate

    def set_downlink_mcs_index(self, downlink_mcs_index):
        self.downlink_mcs_index = downlink_mcs_index

    def set_downlink_mcs_data(self, downlink_mcs_data):
        self.downlink_mcs_data = downlink_mcs_data

    def set_downlink_sinr(self, downlink_sinr):
        self.downlink_sinr = downlink_sinr

    def set_downlink_cqi(self, downlink_cqi):
        self.downlink_cqi = downlink_cqi

    def add_ai_service_subscription(self, ai_service_subscription):
        ai_service_subscription_id = ai_service_subscription.subscription_id
        if ai_service_subscription_id in self.ai_service_subscriptions:
            logger.warning(
                f"UE {self.ue_imsi}: AI service subscription {ai_service_subscription_id} already exists."
            )
        else:
            self.ai_service_subscriptions[ai_service_subscription_id] = (
                ai_service_subscription
            )
            logger.info(
                f"UE {self.ue_imsi}: AI service subscription {ai_service_subscription_id} added."
            )

    def remove_ai_service_subscription(self, ai_service_subscription_id):
        if ai_service_subscription_id in self.ai_service_subscriptions:
            del self.ai_service_subscriptions[ai_service_subscription_id]
            logger.info(
                f"UE {self.ue_imsi}: AI service subscription {ai_service_subscription_id} removed."
            )
        else:
            logger.warning(
                f"UE {self.ue_imsi}: AI service subscription {ai_service_subscription_id} does not exist."
            )

    def cell_selection_and_camping(self):
        # Sort SSBs by received power
        # first sort by frequency priority, then by received power (both the higher the better)
        cells_detected = list(self.downlink_received_power_dBm_dict.values())
        cells_detected.sort(
            key=lambda x: (
                x["frequency_priority"],
                x["received_power_with_cio_dBm"],
            ),
            reverse=True,
        )
        # Print all the detected SSBs in a pretty table
        table_data = [
            [
                v["cell"].cell_id,
                v["received_power_with_cio_dBm"],
                v["frequency_priority"],
            ]
            for v in cells_detected
        ]
        print(f"UE {self.ue_imsi}: Detected SSBs:")
        print(
            tabulate(
                table_data,
                headers=[
                    "Cell ID",
                    "Received Power With CIO (dBm)",
                    "Frequency Priority",
                ],
                tablefmt="grid",
            )
        )

        self.set_current_cell(cells_detected[0]["cell"])
        return True

    def setup_rrc_measurement_event_monitors(self, rrc_meas_events_to_monitor=[]):
        self.rrc_measurement_event_monitors = [
            get_rrc_measurement_event_monitor(event["event_id"], event_params=event)
            for event in rrc_meas_events_to_monitor
        ]

    def authenticate_and_register(self):
        if self.current_bs is None:
            print(
                f"UE {self.ue_imsi}: No base station to authenticate and register with."
            )
            return False

        # simplified one step authentication and registration implementation
        ue_reg_res = self.current_bs.handle_ue_authentication_and_registration(self)
        self.slice_type = ue_reg_res["slice_type"]
        self.qos_profile = ue_reg_res["qos_profile"]
        self.setup_rrc_measurement_event_monitors(ue_reg_res["rrc_meas_events"])
        return True

    def power_up(self):
        print(f"UE {self.ue_imsi} Powering up")
        self.monitor_signal_strength()

        if len(list(self.downlink_received_power_dBm_dict.values())) == 0:
            print(f"UE {self.ue_imsi}: No cells detected. Powering down...")
            return False

        if not self.cell_selection_and_camping():
            print(f"UE {self.ue_imsi}: Cell selection and camping failed.")
            return False

        if not self.authenticate_and_register():
            print(f"UE {self.ue_imsi}: Authentication and registration failed.")
            return False

        self.connected = True

        return True

    def execute_handover(self, target_cell):
        # reset current cell data
        self.downlink_received_power_dBm_dict = {}
        self.set_downlink_sinr(0)
        self.set_downlink_cqi(0)
        self.set_downlink_mcs_index(-1)
        self.set_downlink_mcs_data(None)
        self.set_downlink_bitrate(0)
        self.downlink_latency = 0
        self.uplink_bitrate = 0
        self.uplink_latency = 0
        self.uplink_transmit_power_dBm = settings.UE_TRANSMIT_POWER
        self.set_current_cell(target_cell)

        for event_monitor in self.rrc_measurement_event_monitors:
            event_monitor.reset_trigger_history()

    def set_current_cell(self, cell):
        self.current_cell = cell

        if cell is None:
            if len(self.serving_cell_history) > 0:
                assert (
                    self.serving_cell_history[-1] is not None
                ), f"UE {self.ue_imsi} is not served by any cell."
            self.serving_cell_history.append(None)
        else:
            if len(self.serving_cell_history) > 0:
                assert (
                    self.serving_cell_history[-1] != cell.cell_id
                ), f"UE {self.ue_imsi} is already served by cell {cell.cell_id}."
            self.serving_cell_history.append(cell.cell_id)

        if len(self.serving_cell_history) > settings.UE_SERVING_CELL_HISTORY_LENGTH:
            self.serving_cell_history.pop(0)

    # ---- Trace-driven offered load helpers ----
    def attach_trace(self, samples, speedup: float = 1.0):
        """Attach a list of (t_s, dl_bytes, ul_bytes) samples to replay.
        speedup > 1.0 replays faster; < 1.0 slower.
        """
        if not samples:
            self._trace_samples = None
            self._trace_cursor = 0
            return
        self._trace_samples = samples
        self._trace_cursor = 0
        self._trace_speedup = max(1e-6, float(speedup))
        self._trace_started = True

    def _replay_trace_into_buffer(self, delta_time):
        if not self._trace_samples:
            return
        # Simulation time in seconds
        sim_t = self.simulation_engine.sim_step * delta_time
        # Effective trace time progressed
        target_t = sim_t * self._trace_speedup
        n = len(self._trace_samples)
        while self._trace_cursor < n and self._trace_samples[self._trace_cursor][0] <= target_t:
            _, dl, ul = self._trace_samples[self._trace_cursor]
            if dl > 0:
                self.dl_buffer_bytes += dl
                self.offered_dl_bytes_total += dl
            if ul > 0:
                self.ul_buffer_bytes += ul
            self._trace_cursor += 1

    def deregister(self):
        if self.current_bs is None:
            print(f"UE {self.ue_imsi}: No base station to deregister from.")
            return False
        print(f"UE {self.ue_imsi}: Sending deregistration request.")
        self.current_bs.handle_deregistration_request(self)
        self.set_current_cell(None)
        self.connected = False

    def move_towards_target(self, delta_time):
        if self.target_x is not None and self.target_y is not None:
            dist_to_target = self.dist_to_target
            max_move_dist = self.speed_mps * delta_time
            if dist_to_target <= max_move_dist:
                self.position_x = self.target_x
                self.position_y = self.target_y
            else:
                # move towards the target for the distance of max_move_dist, but round to nearest integer
                ratio = max_move_dist / dist_to_target
                self.position_x += (self.target_x - self.position_x) * ratio
                self.position_y += (self.target_y - self.position_y) * ratio
                self.position_x = round(self.position_x)
                self.position_y = round(self.position_y)

    def monitor_signal_strength(self):
        if self.simulation_engine is None:
            return False

        # monitors the downlink signal strength from the cells

        self.downlink_received_power_dBm_dict = {}
        self.set_downlink_sinr(0)
        self.set_downlink_cqi(0)

        pass_loss_model = settings.CHANNEL_PASS_LOSS_MODEL_MAP[
            settings.CHANNEL_PASS_LOSS_MODEL_URBAN_MACRO_NLOS
        ]
        for cell in self.simulation_engine.cell_list.values():
            # Check if the cell is within the UE's range
            distance = dist_between(
                self.position_x,
                self.position_y,
                cell.position_x,
                cell.position_y,
            )

            received_power_dBm = cell.transmit_power_dBm - pass_loss_model(
                distance_m=distance, frequency_ghz=cell.carrier_frequency_MHz / 1000
            )
            received_power_with_cio_dBm = (
                received_power_dBm + cell.cell_individual_offset_dBm
            )
            if (
                received_power_dBm > settings.UE_SSB_DETECTION_THRESHOLD
                and received_power_dBm >= cell.qrx_level_min
            ):
                self.downlink_received_power_dBm_dict[cell.cell_id] = {
                    "cell": cell,
                    "received_power_dBm": received_power_dBm,
                    "frequency_priority": cell.frequency_priority,
                    "received_power_with_cio_dBm": received_power_with_cio_dBm,
                }
            elif self.current_cell and cell.cell_id == self.current_cell.cell_id:
                # make sure the current cell is in the list of detecte cells
                self.downlink_received_power_dBm_dict[cell.cell_id] = {
                    "cell": cell,
                    "received_power_dBm": settings.UE_SSB_DETECTION_THRESHOLD,
                    "frequency_priority": cell.frequency_priority,
                    "received_power_with_cio_dBm": settings.UE_SSB_DETECTION_THRESHOLD
                    + cell.cell_individual_offset_dBm,
                }

        self.calculate_SINR_and_CQI()

        return True

    def calculate_SINR_and_CQI(self):
        if self.current_cell is None:
            return False

        # make sure the current cell is in the list of detected cells
        power_data = self.downlink_received_power_dBm_dict.get(
            self.current_cell.cell_id, None
        )
        if power_data is None:
            current_cell_power_dBm = self.current_cell.qrx_level_min
        else:
            current_cell_power_dBm = power_data["received_power_dBm"]

        # calculate the SINR
        received_powers_w = np.array(
            [
                dbm_to_watts(cell_power_data["received_power_dBm"])
                for cell_power_data in self.downlink_received_power_dBm_dict.values()
                if cell_power_data["cell"].carrier_frequency_MHz
                == self.current_cell.carrier_frequency_MHz
            ]
        )

        # Serving cell is the one with max received power
        current_cell_power_w = dbm_to_watts(current_cell_power_dBm)
        interference_power_w = np.sum(received_powers_w) - current_cell_power_w

        # Thermal noise
        k = 1.38e-23  # Boltzmann constant
        noise_power_w = k * settings.UE_TEMPERATURE_K * self.current_cell.bandwidth_Hz

        # print(f"UE {self.ue_imsi}: Interference power (W):", interference_power_w)
        # print(f"UE {self.ue_imsi}: Noise power (W):", noise_power_w)
        # print(
        #     f"UE {self.ue_imsi}: Current cell received power: {current_cell_received_power} (dBm) = {current_cell_power_w} (W):"
        # )

        self.set_downlink_sinr(
            10 * np.log10(current_cell_power_w / (interference_power_w + noise_power_w))
        )
        self.set_downlink_cqi(sinr_to_cqi(self.downlink_sinr))
        # print(
        #     f"UE {self.ue_imsi}: Downlink SINR: {self.downlink_sinr:.2f} dB, CQI: {self.downlink_cqi}"
        # )

    def check_rrc_meas_events_to_monitor(self):
        if self.current_bs is None:
            print(
                f"UE {self.ue_imsi}: No base station to report RRC measurement events."
            )
            return False

        cell_signal_map = {
            v["cell"].cell_id: v["received_power_with_cio_dBm"]
            for v in self.downlink_received_power_dBm_dict.values()
        }
        for rrc_meas_event_trigger in self.rrc_measurement_event_monitors:
            rrc_meas_event_trigger.check(self, cell_signal_map.copy())
            if rrc_meas_event_trigger.is_triggered:
                print(
                    f"UE {self.ue_imsi}: RRC measurement event {rrc_meas_event_trigger.event_id} triggered."
                )
                event_report = rrc_meas_event_trigger.gen_event_report()
                # print(f"{self} Reporting event: {event_report}")
                self.current_bs.receive_ue_rrc_meas_events(event_report)

    def request_ai_service(self):
        if self.current_bs is None:
            logger.warning(
                f"UE {self.ue_imsi}: No base station to request AI service from."
            )
            self.ai_service_responses = {}
            return

        if len(self.ai_service_subscriptions) == 0:
            # logger.warning(f"UE {self.ue_imsi}: No AI service subscriptions available.")
            self.ai_service_responses = {}
            return

        if self.downlink_bitrate == 0:
            logger.warning(
                f"UE {self.ue_imsi}: Downlink bitrate is 0, cannot request AI service."
            )
            self.ai_service_responses = {}
            return

        self.ai_service_request_countdonw -= 1
        if self.ai_service_request_countdonw > 0:
            logger.info(
                f"UE {self.ue_imsi}: AI service request countdown: {self.ai_service_request_countdonw}"
            )
            return

        self.ai_service_request_countdonw = settings.UE_AI_SERVICE_REQUEST_COUNTDOWN
        for ai_service_subscription in self.ai_service_subscriptions.values():

            sample_request_data = get_random_ai_service_request_data()
            files, size, name = (
                sample_request_data["files"],
                sample_request_data["size"],
                sample_request_data["name"],
            )

            ai_service_request_data = prepare_ai_service_sample_request(
                ai_service_subscription.ai_service_name, self.ue_imsi, files
            )
            logger.info(
                f"UE {self.ue_imsi}: Requesting AI service {ai_service_subscription.ai_service_name} with {name}."
            )
            start_time_ms = time.time() * 1000  # Convert to milliseconds
            response = self.current_bs.on_ue_application_traffic(
                self, ai_service_request_data
            )
            end_time_ms = time.time() * 1000  # Convert to milliseconds

            # total latency is the time taken to process the request plus the air transmission time.
            # for the moment we use both achivable downlink bitrate to estimate the air transmission time
            ai_service_latency_ms = (
                end_time_ms
                - start_time_ms
                + size * 8 / self.downlink_bitrate * 1000 * 2
            )

            if response is None:
                # Service not deployed/ready or subscription missing for this BS/UE
                logger.warning(
                    f"UE {self.ue_imsi}: AI service '{ai_service_subscription.ai_service_name}' not ready or not deployed; skipping this request cycle."
                )
                continue

            logger.info(
                f"UE {self.ue_imsi}: AI service response: {response.get('response', 'No response field.')}, "
            )
            self.ai_service_responses[ai_service_subscription.subscription_id] = {
                "latency": ai_service_latency_ms,
                "response": response,
                "ai_service_name": ai_service_subscription.ai_service_name,
            }

    def step(self, delta_time):
        if settings.SIM_UE_MOBILITY_ENABLED:
            self.move_towards_target(delta_time)
        # Enqueue trace-driven offered load (if any)
        self._replay_trace_into_buffer(delta_time)
        self.monitor_signal_strength()
        self.check_rrc_meas_events_to_monitor()
        self.request_ai_service()
        self.time_remaining -= delta_time
        if self.time_remaining <= 0:
            self.deregister()

    def to_json(self):
        return {
            "ue_imsi": self.ue_imsi,
            "operation_region": self.operation_region,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "vis_position_x": self.position_x * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "vis_position_y": self.position_y * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "vis_target_x": self.target_x * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "vis_target_y": self.target_y * settings.REAL_LIFE_DISTANCE_MULTIPLIER,
            "speed_mps": self.speed_mps,
            "slice_type": self.slice_type,
            "qos_profile": self.qos_profile,
            "current_cell": self.current_cell.cell_id if self.current_cell else None,
            "current_bs": self.current_bs.bs_id if self.current_bs else None,
            "connected": self.connected,
            "time_remaining": self.time_remaining,
            "serving_cell_history": [cell_id for cell_id in self.serving_cell_history],
            "downlink_bitrate": self.downlink_bitrate,
            "downlink_latency": self.downlink_latency,
            "uplink_bitrate": self.uplink_bitrate,
            "uplink_latency": self.uplink_latency,
            "downlink_received_power_dBm_dict": {
                cell_id: {
                    "received_power_dBm": cell_data["received_power_dBm"],
                    "received_power_with_cio_dBm": cell_data[
                        "received_power_with_cio_dBm"
                    ],
                    "frequency_priority": cell_data["frequency_priority"],
                }
                for cell_id, cell_data in self.downlink_received_power_dBm_dict.items()
            },
            "downlink_sinr": self.downlink_sinr,
            "downlink_cqi": self.downlink_cqi,
            "downlink_mcs_index": self.downlink_mcs_index,
            "downlink_mcs_data": self.downlink_mcs_data,
            "ai_service_subscriptions": {
                subscription_id: subscription.to_json()
                for subscription_id, subscription in self.ai_service_subscriptions.items()
            },
            "ai_service_request_countdonw": self.ai_service_request_countdonw,
            "ai_service_responses": {
                subscription_id: {
                    "latency": response["latency"],
                    "response": response["response"],
                    "ai_service_name": response["ai_service_name"],
                }
                for subscription_id, response in self.ai_service_responses.items()
            },
            "dl_buffer_bytes": self.dl_buffer_bytes,
            "offered_dl_bytes_total": self.offered_dl_bytes_total,
            "served_dl_bytes_total": self.served_dl_bytes_total,
        }
