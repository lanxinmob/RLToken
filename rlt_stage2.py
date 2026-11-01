"""ARX transport for upstream RLT Stage 2; no SDK imports in this process."""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import ExitStack
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import numpy as np
import zmq


ROOT = Path(__file__).resolve().parent
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7)) + ("gripper",)
PROFILE = "rlt_arx5_stage2_v1"
HARDWARE_KEYS = {
    "single_arm",
    "follower_model", "follower_interface", "follower_gripper_open_readout", "follower_gravity_scale",
    "leader_model", "leader_interface", "leader_gripper_open_readout", "leader_gravity_scale",
    "control_hz", "action_hz", "joint_speed", "gripper_speed", "mirror_gain_scale", "drag_kd_scale",
    "drag_gripper_kd", "gripper_close_threshold", "feedback_max_age_s", "camera_max_age_s",
    "tracking_error_max_rad", "lease_timeout_s", "action_timeout_s", "observation_timeout_s",
    "rpc_timeout_s", "startup_timeout_s", "chunk_filter", "chunk_ema_alpha",
}


def gripper_check(args):
    """Query DM motor 8 on the two ARX buses; never enable, move, or zero it."""
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    config = json.loads(args.config.read_text())["hardware_config"]
    hardware_config = config
    single_arm = bool(hardware_config.get("single_arm", False))
    if single_arm:
        arm_specs = (("follower", hardware_config["follower_model"], 0.086),)
    else:
        arm_specs = (("leader", "X5", 0.088), ("follower", "X5_umi", 0.086))
    arms = []
    for side, model, width in arm_specs:
        interface = config[f"{side}_interface"]
        readout = config[f"{side}_gripper_open_readout"]
        if config[f"{side}_model"] != model or not isinstance(interface, str) or not interface:
            raise ValueError(f"Expected {side} {model} with a CAN interface")
        if isinstance(readout, bool) or not isinstance(readout, (int, float)) or not math.isfinite(readout) or readout >= 0:
            raise ValueError(f"Invalid {side} gripper calibration")
        arms.append(dict(side=side, interface=interface, open_readout_rad=readout, width_m=width))
    interfaces = {arm["interface"] for arm in arms}
    if not single_arm and len(interfaces) != 2:
        raise ValueError("Leader and follower require distinct CAN interfaces")
    # Standard DM status refresh (local Damiao driver _refresh_motor).
    # The only packet transmitted by this diagnostic is a status request.
    request = struct.pack("=IB3x8s", 0x7FF, 8, bytes.fromhex("0800cc0000000000"))
    summary = {arm["side"]: dict(interface=arm["interface"], samples=0, min_angle_rad=None,
                                 max_angle_rad=None, last=None, range_violations=0) for arm in arms}

    def display(elapsed):
        messages = []
        for side, state in summary.items():
            last = state["last"]
            prefix = f"{side}/{state['interface']}"
            if last is None:
                messages.append(f"{prefix}: no motor 8 reply")
            else:
                span = state["max_angle_rad"] - state["min_angle_rad"]
                messages.append(f"{prefix}: angle_rad={last['angle_rad']:.6f} "
                                f"scaled_mm={last['scaled_width_mm']:.3f} span_rad={span:.6f} "
                                f"samples={state['samples']} age_s={elapsed-last['elapsed_s']:.3f}")
        print(" | ".join(messages), flush=True)

    interrupted = False
    with ExitStack() as stack:
        for arm in arms:
            lock = stack.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM))
            lock.bind("\0rlt-arx-" + arm["interface"])
        # Detect existing SocketCAN readers before opening our CAN sockets.
        for table in Path("/proc/net/can").glob("rcvlist_*"):
            for line in table.read_text().splitlines():
                fields = line.split()
                if fields and fields[0] in interfaces | {"(any)"} and "no entry" not in line:
                    raise RuntimeError(f"CAN already in use; stop the existing program first: {line.strip()}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log = stack.enter_context(args.output.open("x"))
        log.write(json.dumps(dict(kind="start", wall_time=time.time(), arms=arms,
                                  request_can_id=0x7FF, request_data="0800cc0000000000")) + "\n")
        buses = {}
        for arm in arms:
            bus = stack.enter_context(socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW))
            bus.bind((arm["interface"],))
            bus.setblocking(False)
            buses[bus] = arm
        prompt = ("Read-only gripper feedback: open/close the configured execution arm."
                  if single_arm else
                  "Read-only gripper feedback: open/close the leader, then the follower, to identify each bus.")
        print(f"{prompt}\nLogging to {args.output}", flush=True)
        start = time.monotonic()
        next_request, next_display = start, start + 1
        try:
            while time.monotonic() - start < args.seconds:
                now = time.monotonic()
                if now >= next_request:
                    for bus in buses:
                        bus.send(request)
                    next_request = now + 0.1
                ready, _, _ = select.select(list(buses), [], [], max(0, min(next_request, start + args.seconds) - time.monotonic()))
                for bus in ready:
                    can_id, size, data = struct.unpack("=IB3x8s", bus.recv(16))
                    arm = buses[bus]
                    elapsed = time.monotonic() - start
                    record = dict(kind="feedback", elapsed_s=elapsed, side=arm["side"],
                                  interface=arm["interface"], can_id=can_id, dlc=size, data=data[:size].hex())
                    # This ARX profile uses master CAN ID 0; byte 0 contains motor ID/state.
                    if can_id == 0 and size == 8 and data[0] & 0xF == 8:
                        angle = int.from_bytes(data[1:3], "big") * 25 / 65535 - 12.5
                        width = angle / arm["open_readout_rad"] * arm["width_m"]
                        record.update(angle_rad=angle, scaled_width_mm=width * 1000, motor_state=data[0] >> 4)
                        state = summary[arm["side"]]
                        state["samples"] += 1
                        state["min_angle_rad"] = angle if state["min_angle_rad"] is None else min(state["min_angle_rad"], angle)
                        state["max_angle_rad"] = angle if state["max_angle_rad"] is None else max(state["max_angle_rad"], angle)
                        state["last"] = record
                        state["range_violations"] += int(width < -0.005 or width > arm["width_m"] + 0.005)
                    log.write(json.dumps(record) + "\n")
                if time.monotonic() >= next_display:
                    display(time.monotonic() - start)
                    log.flush()
                    next_display = time.monotonic() + 1
        except KeyboardInterrupt:
            interrupted = True
        finally:
            log.write(json.dumps(dict(kind="summary", diagnostic_only=True, interrupted=interrupted,
                                      elapsed_s=time.monotonic() - start, arms=summary)) + "\n")
            display(time.monotonic() - start)
    if not interrupted and any(state["samples"] == 0 for state in summary.values()):
        raise RuntimeError("Missing motor 8 feedback; see the raw log")


def vector(values, name="action"):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 7D vector (joint rad, gripper m)")
    return result


def action_dict(values):
    return {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, vector(values))}


def action_vector(values):
    if not isinstance(values, dict) or set(values) != {f"{name}.pos" for name in JOINT_NAMES}:
        raise ValueError("ARX actions must contain joint_1.pos through joint_6.pos and gripper.pos")
    return vector([values[f"{name}.pos"] for name in JOINT_NAMES])


def validate_config(config, args, rlt_config):
    if not isinstance(config, dict):
        raise ValueError("hardware_config must be a JSON object")
    # Keep older dual-arm JSON files usable; absence of the new mode flag
    # means the original distinct follower/leader contract.
    if "single_arm" not in config:
        config = {"single_arm": False, **config}
    if set(config) != HARDWARE_KEYS:
        raise ValueError(f"ARX hardware config keys differ: missing={sorted(HARDWARE_KEYS-set(config))}, unknown={sorted(set(config)-HARDWARE_KEYS)}")
    if rlt_config.state_dim != 7 or rlt_config.action_dim != 7:
        raise ValueError("ARX Stage 2 requires exactly 7D state/action features")
    if not isinstance(config["single_arm"], bool):
        raise ValueError("single_arm must be a boolean")
    if config["single_arm"]:
        if config["follower_model"] not in {"X5", "X5_umi"}:
            raise ValueError("Single-arm ARX profile requires follower X5 or X5_umi")
        if config["leader_model"] != config["follower_model"]:
            raise ValueError("Single-arm ARX profile requires leader_model to match follower_model")
        if config["leader_interface"] != config["follower_interface"]:
            raise ValueError("Single-arm ARX profile requires one shared CAN interface")
        if config["leader_gripper_open_readout"] != config["follower_gripper_open_readout"]:
            raise ValueError("Single-arm ARX profile requires one shared gripper calibration")
    else:
        if config["follower_model"] != "X5_umi" or config["leader_model"] != "X5":
            raise ValueError("This ARX profile requires follower X5_umi and leader X5")
        if config["follower_interface"] == config["leader_interface"]:
            raise ValueError("Leader and follower require distinct CAN interfaces")
    if any(not isinstance(config[f"{side}_interface"], str) or not config[f"{side}_interface"]
           for side in ("follower", "leader")):
        raise ValueError("ARX CAN interfaces must be non-empty strings")
    for key, value in config.items():
        if key in {"single_arm", "follower_model", "leader_model", "follower_interface", "leader_interface", "chunk_filter"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if key == "gripper_close_threshold":
            if not 0 <= value <= 0.005:
                raise ValueError("gripper_close_threshold must be between 0 and 0.005 metres")
        elif "gripper_open_readout" in key:
            if value >= 0:
                raise ValueError(f"{key} must be negative for the calibrated ARX profile")
        elif value <= 0:
            raise ValueError(f"{key} must be positive")
    if config["chunk_filter"] != "none":
        raise ValueError("ARX Stage 2 requires chunk_filter='none'; RTC owns chunk continuity")
    if config["chunk_ema_alpha"] > 1:
        raise ValueError("chunk_ema_alpha must be at most 1")
    if config["joint_speed"] > 3.6:
        raise ValueError("ARX Stage 2 joint_speed must be at most 3.6 rad/s (0.12 rad at 30 Hz)")
    if config["tracking_error_max_rad"] > 0.15:
        raise ValueError("ARX Stage 2 tracking_error_max_rad must be at most 0.15 rad")
    if config["mirror_gain_scale"] > 1 or config["drag_kd_scale"] > 1:
        raise ValueError("ARX gain scales must be at most 1")
    if config["action_hz"] != 30 or args.control_fps != 30 or config["control_hz"] < 30:
        raise ValueError("Dataset/replay use 30 Hz; SDK monitoring must run at least at 30 Hz")
    if config["rpc_timeout_s"] >= config["lease_timeout_s"]:
        raise ValueError("RPC timeout must be shorter than the worker lease")
    if args.reset_between_episodes or not args.manual_positioning:
        raise ValueError("ARX Stage 2 requires manual positioning and no automatic reset")
    if args.rtc_enabled:
        if getattr(args, "rtc_mode", "guided") != "guided":
            raise ValueError("ARX Stage 2 requires RTC mode='guided' with the frozen pi0.5 checkpoint")
        if args.rtc_execution_horizon != rlt_config.n_action_steps_rl:
            raise ValueError(
                "ARX Stage 2 RTC horizon must equal the RLT execution horizon "
                f"({rlt_config.n_action_steps_rl})"
            )
    if args.pad_missing_cameras or args.camera_names != ["cam_wrist"]:
        raise ValueError("ARX requires its real cam_wrist stream; camera padding is unsupported")
    warmup_s = json.loads(args.cameras)["cam_wrist"].get("warmup_s", 3)
    if isinstance(warmup_s, bool) or not isinstance(warmup_s, (int, float)) or not math.isfinite(warmup_s) or warmup_s <= 0:
        raise ValueError("ARX camera warmup_s must be positive and finite")
    if 2 * warmup_s >= config["startup_timeout_s"]:
        raise ValueError("ARX startup_timeout_s must allow both camera warmups")
    if args.max_relative_target is not None or args.actor_max_relative_target is not None:
        raise ValueError("Use ARX rad/s and m/s limits in hardware_config instead of SO-101 degree limits")
    return dict(config)


class Follower:
    def __init__(self, runtime):
        self.runtime = runtime

    def get_observation(self):
        observation = self.runtime.call("OBSERVATION")
        result = action_dict(observation["follower"]["position"])
        result["cam_wrist"] = observation["image_rgb"]
        result["_rlt_execution_state"] = np.asarray(observation["execution_state"], dtype=np.float32)
        return result

    def send_action(self, action):
        return action_dict(self.runtime.send_target(action_vector(action))["accepted"])

    def disconnect(self):
        self.runtime.close()


class Leader:
    joint_names = JOINT_NAMES

    def __init__(self, runtime):
        self.runtime = runtime

    def get_action(self):
        state = self.runtime.call("LEADER")
        values = vector(state["position"]).copy()
        # Match LeaderFollowerTeleop._follower_gripper_target: metres, a 1 mm
        # close threshold, and the follower's physical width as the upper bound.
        values[-1] = 0.0 if values[-1] <= self.runtime.config["gripper_close_threshold"] else min(values[-1], 0.086)
        return action_dict(values)

    def disconnect(self):
        self.runtime.close()


class Runtime:
    """Own a worker lease and use one short-lived REQ socket per RPC.

    Socket recreation avoids REQ's invalid state after timeout. Separate sockets
    let emergency stop and heartbeats proceed while the inference thread waits.
    """

    joint_names = JOINT_NAMES
    action_units = "rad/m"

    def __init__(self, config, args, output_dir, should_stop, on_fault):
        self.config = config
        self.single_arm = bool(config.get("single_arm", False))
        self.args = args
        self.output_dir = Path(output_dir)
        self.should_stop = should_stop
        self.on_fault = on_fault
        self.context = zmq.Context()
        self.worker = None
        self.token = uuid.uuid4().hex
        self.address = None
        self.fault = None
        self.phase = "idle"
        self._stop = threading.Event()
        self._halted = threading.Event()
        self._thread = None
        self._log = None
        self._worker_log = None
        self._transport_dir = None
        self._log_lock = threading.Lock()
        self.follower = Follower(self)
        self.leader = Leader(self)

    def connect(self):
        config = json.loads((ROOT / "config.json").read_text())
        self._transport_dir = tempfile.TemporaryDirectory(prefix="rlt-arx-")
        self.address = f"ipc://{self._transport_dir.name}/worker.sock"
        self._worker_log = (self.output_dir / "hardware_worker.log").open("w", encoding="utf-8")
        self._log = (self.output_dir / "hardware_actions.jsonl").open("w", encoding="utf-8")
        worker_config = {
            **self.config,
            "camera": json.loads(self.args.cameras)["cam_wrist"],
            "preview": bool(getattr(self.args, "preview", False)),
        }
        self.worker = subprocess.Popen(
            [config["sdk_python"], "-B", "-u", str(ROOT / "rlt_stage2_hardware.py"),
             "--address", self.address, "--token", self.token, "--config", json.dumps(worker_config)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=self._worker_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.config["startup_timeout_s"]
        try:
            while True:
                if self.worker.poll() is not None:
                    raise RuntimeError(self._startup_error(f"ARX worker exited {self.worker.returncode}"))
                try:
                    info = self.call("INFO", latch=False)
                    break
                except TimeoutError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(self._startup_error("ARX worker startup timed out"))
            if info["profile"] != PROFILE:
                raise ValueError(f"Unexpected ARX worker profile: {info}")
            widths = (("follower", 0.086), ("leader", 0.086)) if self.single_arm else (
                ("follower", 0.086), ("leader", 0.088),
            )
            for side, width in widths:
                actual = info[side]
                if actual["model"] != self.config[f"{side}_model"] or actual["interface"] != self.config[f"{side}_interface"]:
                    raise ValueError(f"ARX {side} model/CAN differs from requested configuration")
                if actual["joint_dof"] != 6 or abs(actual["gripper_width"] - width) > 1e-6:
                    raise ValueError(f"ARX {side} DOF/gripper width differs from the dataset contract")
                if abs(actual["gripper_open_readout"] - self.config[f"{side}_gripper_open_readout"]) > 1e-6:
                    raise ValueError(f"ARX {side} calibration differs from requested configuration")
            (self.output_dir / "hardware_info.json").write_text(json.dumps(info, indent=2) + "\n")
            self.execution_transform_profile = info["execution_transform"]
            self._thread = threading.Thread(target=self._supervise, name="arx-lease", daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def _startup_error(self, reason):
        path = self.output_dir / "hardware_worker.log"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            # SDK faults precede controller cleanup and the Python traceback.
            # Keep their first causes even when they fall outside the log tail.
            sdk_errors = [line for line in lines[:-8] if " error]" in line]
            tail = "\n".join(sdk_errors[:4] + lines[-8:])
        except OSError as exc:
            tail = f"Could not read hardware log: {exc}"
        return f"{reason}; hardware log: {path}\n{tail}"

    def call(self, command, data=None, *, latch=True, timeout_s=None):
        if self.fault and command not in {"STOP", "SHUTDOWN", "INFO", "RETURN_HOME"}:
            raise RuntimeError(f"ARX session fault is latched: {self.fault}")
        socket = self.context.socket(zmq.REQ)
        if timeout_s is None:
            timeout_s = self.config["rpc_timeout_s"]
        timeout_ms = max(1, int(timeout_s * 1000))
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.address)
        try:
            socket.send_pyobj({"cmd": command, "token": self.token, "data": data})
            reply = socket.recv_pyobj()
            if not isinstance(reply, dict) or reply.get("ok") is not True:
                raise RuntimeError(f"ARX {command} rejected: {reply}")
            return reply["data"]
        except zmq.Again as exc:
            if latch:
                self.fault = f"{command} RPC timed out"
            raise TimeoutError(f"ARX {command} RPC timed out") from exc
        except Exception as exc:
            if latch:
                self.fault = str(exc)
            raise
        finally:
            socket.close(0)

    def _supervise(self):
        while not self._stop.wait(0.05):
            if self._halted.is_set():
                return
            try:
                if self.should_stop() or self.fault:
                    if self.fault:
                        self.on_fault(self.fault)
                    self.call("STOP")
                    return
                self.call("HEARTBEAT")
            except Exception as exc:
                self.fault = str(exc)
                self.on_fault(self.fault)
                try:
                    self.call("STOP")
                except Exception:
                    pass
                return

    def set_phase(self, phase, *, refresh=False):
        if phase == self.phase and not refresh:
            return
        result = self.call("PHASE", phase)
        self.phase = phase
        if phase == "human" and result["gripper_pickup_width_m"] is not None:
            logging.getLogger(__name__).info(
                "Human gripper pickup: follower holds %.1f mm; move the leader gripper to the same width to begin following.",
                result["gripper_pickup_width_m"] * 1000,
            )
        return result

    def position_tick(self):
        return self.call("POSITION_TICK")

    def update_leader(self, human, target):
        # The worker owns the actual 100 Hz mirror/drag control, including its
        # watchdog. This RPC changes modes, never simulates SO-101 registers.
        human = human or self.phase in {"human", "position"}
        return self.call("LEADER_MODE", "drag" if human else "mirror")

    def send_target(self, target):
        try:
            receipt = self.call("TARGET", vector(target).tolist())
            vector(receipt["accepted"], "SDK accepted target")
            if receipt["gripper_pickup_completed"]:
                logging.getLogger(__name__).info("Human gripper matched; follower now tracks leader opening and closing.")
            with self._log_lock:
                self._log.write(json.dumps({"monotonic": time.monotonic(), "phase": self.phase,
                                           "requested": target.tolist(), **receipt}) + "\n")
                self._log.flush()
            return receipt
        except Exception as exc:
            # A lost acknowledgement or full log disk must not cause the caller
            # to retry a movement that the SDK may already have executed.
            self.fault = str(exc)
            self.on_fault(self.fault)
            try:
                self.stop()
            except Exception as stop_exc:
                self.fault = f"{self.fault}; stop failed: {stop_exc}"
            raise

    def return_home(self):
        """Run the SDK worker's bounded home trajectory and wait for its reply."""
        self._halted.set()
        # Home movement is deliberately slow and the SDK waits for feedback
        # before replying.  The normal RPC timeout protects the 30 Hz policy
        # loop, but is too short for this bounded trajectory.
        result = self.call("RETURN_HOME", timeout_s=max(self.config["startup_timeout_s"], 30.0))
        self.phase = "idle"
        self.fault = None
        return result

    def stop(self):
        self._halted.set()
        self.phase = "stopped"
        if self.worker is not None and self.worker.poll() is None:
            self.call("STOP")

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._halted.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        if self.worker is not None:
            try:
                if self.worker.poll() is None:
                    self.call("SHUTDOWN")
            except Exception:
                pass
            try:
                self.worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait()
            self.worker = None
        self.context.term()
        if self._transport_dir is not None:
            self._transport_dir.cleanup()
        for handle in (self._log, self._worker_log):
            if handle is not None:
                handle.close()


def run(args):
    """Translate the small local CLI into the pinned upstream Stage 2 entry."""
    checkpoint = args.checkpoint.resolve()
    if (checkpoint / "pretrained_model").is_dir():
        checkpoint /= "pretrained_model"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "lerobot/src")])
    command = [sys.executable, "-B", str(ROOT / "scripts/train_rlt_stage2_pi05.py"),
               "--config", str(args.config.resolve()), "--pi05_path", str(checkpoint),
               "--rlt_checkpoint", str(args.rlt_checkpoint.resolve()),
               "--output_dir", str(args.output.resolve()), "--max_episodes", str(args.episodes)]
    if args.dry_run:
        command.append("--dry_run")
    if getattr(args, "resume_from", None) is not None:
        command.extend(["--resume_from", str(args.resume_from.resolve())])
    if getattr(args, "auto_critical", False):
        command.append("--auto_critical")
    if getattr(args, "async_inference", False):
        command.append("--async_inference")
    if not getattr(args, "rtc_enabled", True):
        command.append("--no-rtc_enabled")
    if getattr(args, "rtc_mode", None) is not None:
        command.extend(["--rtc_mode", args.rtc_mode])
    if getattr(args, "rtc_max_guidance_weight", None) is not None:
        command.extend(["--rtc_max_guidance_weight", str(args.rtc_max_guidance_weight)])
    if getattr(args, "rtc_execution_horizon", None) is not None:
        command.extend(["--rtc_execution_horizon", str(args.rtc_execution_horizon)])
    if getattr(args, "actor_execute_mean", False):
        command.append("--actor_execute_mean")
    if getattr(args, "start_blend_steps", None) is not None:
        command.extend(["--start_blend_steps", str(args.start_blend_steps)])
    if getattr(args, "vla_only", False):
        command.append("--vla_only")
    if getattr(args, "exec_grid_hz", None) is not None:
        command.extend(["--exec_grid_hz", str(args.exec_grid_hz)])
    if getattr(args, "critical_classifier", None) is not None:
        command.extend(["--critical_classifier", str(args.critical_classifier.resolve())])
    if getattr(args, "steps_per_episode", None) is not None:
        if args.steps_per_episode < 10:
            raise ValueError("--steps-per-episode must be at least one 10-step control window")
        command.extend(["--steps_per_episode", str(args.steps_per_episode)])
    if not args.dry_run and not getattr(args, "no_preview", False):
        command.append("--preview")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
