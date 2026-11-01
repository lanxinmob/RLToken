#!/usr/bin/env python3
"""ARX SDK worker for RLT Stage 2, supporting dual-arm and single-arm HIL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import json
import signal
import socket as unix_socket
import threading
import time

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from rlt_action_transform import EXECUTION_CONTRACT, accepted_target

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pi05"))
import pi05_hardware as hardware
from recording.hdf5_recorder import check_camera_streams


PROFILE = "rlt_arx5_stage2_v1"


def target_vector(value):
    return hardware.finite_array(value, (7,), "ARX target")


class Arm(hardware.SingleArmWorker):
    """Reuse VLA-only SDK state/target validation with explicit mode changes."""

    def __init__(self, config, side):
        self.args = argparse.Namespace(
            model=config[f"{side}_model"], interface=config[f"{side}_interface"],
            joint_speed=config["joint_speed"], gripper_speed=config["gripper_speed"],
            drag_kd_scale=config["drag_kd_scale"], drag_gripper_kd=config["drag_gripper_kd"],
        )
        self.side = side
        self.config = config
        self.camera = None
        self.controller = None
        self.mode = "damped"
        self.last_target = None
        self.last_command_time = None
        sdk = hardware.sdk
        self.robot = sdk.RobotConfigFactory.get_instance().get_config(self.args.model)
        self.robot.gripper_open_readout = config[f"{side}_gripper_open_readout"]
        self.robot.gravity_vector = self.robot.gravity_vector * config[f"{side}_gravity_scale"]
        self.robot.joint_vel_max = np.minimum(self.robot.joint_vel_max, self.args.joint_speed)
        self.robot.gripper_vel_max = min(self.robot.gripper_vel_max, self.args.gripper_speed)
        self.control = sdk.ControllerConfigFactory.get_instance().get_config("joint_controller", self.robot.joint_dof)
        self.control.gravity_compensation = True
        self.control.default_preview_time = 0.0
        if self.robot.joint_dof != 6:
            raise ValueError("ARX Stage 2 requires six arm joints")
        expected_width = 0.086 if side == "follower" else 0.088
        if abs(self.robot.gripper_width - expected_width) > 1e-6:
            raise ValueError(f"Unexpected {side} gripper width: {self.robot.gripper_width}")

    def connect(self):
        self.controller = hardware.sdk.Arx5JointController(self.robot, self.control, self.args.interface)
        self.set_mode("hold")

    def fresh_state(self):
        state = self._state()
        age = float(state["feedback_age_s"])
        if state["emergency"] or not np.isfinite(age) or not 0 <= age <= self.config["feedback_max_age_s"]:
            raise ValueError(f"{self.side} emergency or stale feedback: age={age}")
        state["position"] = np.r_[state["joint_pos"], np.clip(state["gripper_pos"], 0, self.robot.gripper_width)]
        return state

    def set_mode(self, mode):
        if mode == self.mode:
            return self.fresh_state()
        state = self.fresh_state()
        # timestamp=0 cancels the SDK interpolator's old trajectory atomically.
        self._send(state["position"][:6], state["position"][-1], timestamp=0)
        sdk = hardware.sdk
        if mode == "drag":
            gain = sdk.Gain(6)
            gain.kp()[:] = 0
            gain.kd()[:] = self.control.default_kd * self.config["drag_kd_scale"]
            gain.gripper_kp = 0
            gain.gripper_kd = self.config["drag_gripper_kd"]
        elif mode in {"hold", "mirror"}:
            scale = self.config["mirror_gain_scale"] if mode == "mirror" else 1.0
            gain = sdk.Gain(self.control.default_kp * scale, self.control.default_kd,
                            self.control.default_gripper_kp * scale, self.control.default_gripper_kd)
        else:
            raise ValueError(f"Unknown ARX arm mode: {mode}")
        self.controller.set_gain(gain)
        self.mode = mode
        return self.fresh_state()

    def accept(self, desired, dt, *, preview_s, pre_limited=False):
        if self.mode not in {"hold", "mirror"}:
            raise ValueError(f"{self.side} targets are forbidden in {self.mode} mode")
        desired = target_vector(desired)
        measured = self.fresh_state()["position"]
        previous = measured if self.last_target is None else self.last_target
        if np.max(np.abs(previous[:6] - measured[:6])) > self.config["tracking_error_max_rad"]:
            raise ValueError(f"{self.side} tracking error exceeds the configured limit")
        limits = np.r_[np.minimum(self.robot.joint_vel_max, self.args.joint_speed),
                        min(self.robot.gripper_vel_max, self.args.gripper_speed)]
        dt = min(max(float(dt), 0.0), 0.1)
        if pre_limited:
            # Policy phase already applied accepted_target() with the same
            # max_step; a second rate clamp would duplicate the same filter.
            accepted = np.asarray(desired, dtype=np.float64).copy()
        else:
            accepted = previous + np.clip(desired - previous, -limits * dt, limits * dt)
        accepted[:6] = np.clip(accepted[:6], self.robot.joint_pos_min, self.robot.joint_pos_max)
        accepted[-1] = np.clip(accepted[-1], 0, self.robot.gripper_width)
        # Return a receipt only after the SDK has accepted exactly this target.
        self._send(accepted[:6], accepted[-1], timestamp=self.controller.get_timestamp() + preview_s)
        return accepted.copy(), measured.copy()

    def description(self):
        return {"model": self.args.model, "interface": self.args.interface, "joint_dof": int(self.robot.joint_dof),
                "joint_min": self.robot.joint_pos_min.tolist(), "joint_max": self.robot.joint_pos_max.tolist(),
                "gripper_width": float(self.robot.gripper_width),
                "gripper_open_readout": float(self.robot.gripper_open_readout),
                "gravity_scale": self.config[f"{self.side}_gravity_scale"],
                "sdk_controller_dt_s": float(self.control.controller_dt)}

    def return_home(self):
        """Use the SDK's bounded joint-space home trajectory and wait for feedback."""
        state = self.fresh_state()
        if state["emergency"]:
            raise ValueError(f"{self.side} emergency is active; cannot return home")
        robot = self.robot
        travel_time = max(
            float(np.max(np.abs(state["joint_pos"]) / robot.joint_vel_max)),
            abs(float(state["gripper_pos"]) - robot.gripper_width) / robot.gripper_vel_max,
        )
        self.controller.reset_to_home()
        deadline = time.monotonic() + max(3.0, travel_time + 2.0)
        while time.monotonic() < deadline:
            state = self.fresh_state()
            if state["emergency"]:
                raise ValueError(f"SDK emergency during {self.side} home return")
            if (np.max(np.abs(state["joint_pos"])) <= 0.03
                    and abs(float(state["gripper_pos"]) - robot.gripper_width) <= 0.003):
                self.set_mode("hold")
                return state
            time.sleep(0.01)
        raise TimeoutError(f"{self.side} home feedback timed out (joint 0.03 rad, gripper 3 mm)")


class DualArmWorker:
    def __init__(self, config, token):
        self.config = config
        self.single_arm = bool(config.get("single_arm", False))
        self.token = token
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.thread = None
        self.camera = None
        self.follower = None
        self.leader = None
        self.interface_locks = []
        self._cv2 = None
        self._preview_enabled = False
        self._preview_sequence = -1
        self.fault = None
        self.phase = "idle"
        self.leader_mode = "hold"
        self.sequence = 0
        self.filtered = None
        self.gripper_pickup_width = None
        self.gripper_pickup_previous = None
        now = time.monotonic()
        self.last_contact = now
        self.last_observation = now
        self.last_action = now
        self.last_mirror = now
        try:
            if not self.single_arm and config["leader_interface"] == config["follower_interface"]:
                raise ValueError("Leader and follower must use different CAN interfaces")
            sides = ("follower",) if self.single_arm else ("follower", "leader")
            for side in sides:
                # Linux abstract sockets release their lock automatically on exit.
                lock = unix_socket.socket(unix_socket.AF_UNIX, unix_socket.SOCK_DGRAM)
                self.interface_locks.append(lock)
                lock.bind("\0rlt-arx-" + config[f"{side}_interface"])
            camera = config["camera"]
            self.camera = hardware.Camera(camera["index_or_path"], width=camera["width"], height=camera["height"],
                                          fps=camera["fps"], fourcc=camera["fourcc"], output_res=(256, 256))
            self.warmup_camera()
            # Resolve BOTH configs/calibrations before either controller starts.
            self.follower = Arm(config, "follower")
            # In single-arm HIL the policy and operator share one physical
            # controller.  Keep one Arm object so there is no duplicate CAN
            # reader/writer and switch that controller between hold and drag.
            self.leader = self.follower if self.single_arm else Arm(config, "leader")
            self.follower.connect()
            if not self.single_arm:
                self.leader.connect()
            # SDK constructors hold the GIL while initializing CAN. The capture
            # thread cannot publish frames during that interval (~300 ms for
            # two arms), so wait for it to resume before declaring readiness.
            self.warmup_camera(connected=True)
            now = time.monotonic()
            self.last_contact = self.last_observation = self.last_action = self.last_mirror = now
            self._start_preview()
            self.thread = threading.Thread(target=self._watchdog, name="arx-control", daemon=True)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def warmup_camera(self, *, connected=False):
        warmup_s = self.config["camera"].get("warmup_s", 3)
        deadline = time.monotonic() + warmup_s
        status = "both arms hold their current measured pose" if connected else "motors are not connected"
        print(f"Waiting up to {warmup_s:g}s for valid new wrist camera frames from "
              f"{self.config['camera']['index_or_path']}; {status}", flush=True)

        def check_arms():
            if connected:
                self.follower.fresh_state()
                self.leader.fresh_state()

        if connected:
            # Ignore only the startup cache: require a frame received AFTER the
            # constructors returned. sleep() lets the camera thread run again.
            resumed_at = time.monotonic()
            _, _, baseline_sequence = self.camera.latest()
            while True:
                check_arms()
                _, received, sequence = self.camera.latest()
                if sequence > baseline_sequence and received >= resumed_at:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("ARX camera did not resume after controller initialization")
                time.sleep(0.01)
        # Capture cards can emit blank frames while acquiring HDMI sync. Require
        # consecutive new valid frames both before and after SDK initialization.
        check_camera_streams({"cam_wrist": self.camera}, max_age=self.config["camera_max_age_s"],
                             timeout=max(0, deadline - time.monotonic()), required_frames=3, check=check_arms)
        self.fresh_camera()

    def fresh_camera(self):
        rgb, received, sequence = self.camera.latest()
        age = time.monotonic() - received
        if not np.isfinite(age) or not 0 <= age <= self.config["camera_max_age_s"]:
            raise ValueError(f"ARX camera is stale: {age:.3f}s")
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError("ARX camera must deliver RGB uint8 256x256x3")
        if int(rgb.max()) <= 2 or (int(rgb.max()) <= 32 and int(rgb.max()) - int(rgb.min()) <= 2):
            raise ValueError(f"ARX camera has no valid input signal (pixel range {int(rgb.min())}..{int(rgb.max())}); "
                             f"check camera input/channel: {self.config['camera']['index_or_path']}")
        return rgb, float(received), int(sequence), float(age)

    def _start_preview(self):
        """Open the local camera window when a desktop display is available."""
        if not self.config.get("preview", False):
            return
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("ARX live preview disabled: no desktop display is available", flush=True)
            return
        try:
            import cv2

            cv2.namedWindow("RLT ARX5 | cam_wrist", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("RLT ARX5 | cam_wrist", 768, 768)
            self._cv2 = cv2
            self._preview_enabled = True
            print("ARX live preview enabled: RLT ARX5 | cam_wrist (q/Esc closes the window)", flush=True)
        except Exception as exc:
            print(f"ARX live preview disabled: {exc}", flush=True)
            self._cv2 = None

    def _stop_preview(self):
        if self._cv2 is None:
            return
        self._preview_enabled = False
        try:
            self._cv2.destroyWindow("RLT ARX5 | cam_wrist")
        except Exception:
            pass
        self._cv2 = None

    def preview_tick(self):
        """Refresh the GUI from the newest camera frame without touching control state."""
        if not self._preview_enabled or self._cv2 is None or self.camera is None:
            return
        try:
            rgb, _, sequence = self.camera.latest()
            if sequence == self._preview_sequence:
                self._cv2.waitKey(1)
                return
            if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                return
            frame = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
            self._cv2.putText(frame, "cam_wrist", (8, 22), self._cv2.FONT_HERSHEY_SIMPLEX,
                              0.55, (0, 255, 0), 1, self._cv2.LINE_AA)
            self._cv2.imshow("RLT ARX5 | cam_wrist", frame)
            self._preview_sequence = sequence
            key = self._cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._stop_preview()
            elif self._cv2.getWindowProperty("RLT ARX5 | cam_wrist", self._cv2.WND_PROP_VISIBLE) < 1:
                self._stop_preview()
        except Exception as exc:
            print(f"ARX live preview stopped: {exc}", flush=True)
            self._stop_preview()

    def stop(self, reason=None):
        # Try both arms even if the first SDK call fails.
        errors = []
        if reason is not None and self.fault is None:
            self.fault = str(reason)
        self.phase = "stopped"
        self.filtered = None
        arms = (self.follower,) if self.single_arm else (self.follower, self.leader)
        for arm in arms:
            if arm is not None and arm.controller is not None:
                try:
                    arm.stop()
                except Exception as exc:
                    errors.append(str(exc))
        if errors:
            self.fault = f"Stopping ARX failed: {errors}"
            raise RuntimeError(self.fault)
        return {"stopped": True, "fault": self.fault}

    def check_health(self, now=None):
        now = time.monotonic() if now is None else now
        if self.phase == "stopped":
            return
        if now - self.last_contact > self.config["lease_timeout_s"]:
            raise TimeoutError("ARX owner lease expired")
        self.follower.fresh_state()
        self.leader.fresh_state()
        self.fresh_camera()
        if self.phase != "idle":
            if now - self.last_action > self.config["action_timeout_s"]:
                raise TimeoutError("ARX action/inference timeout")
            # Manual control uses fresh arm feedback, not policy observations.
            # Its action timeout and the camera/arm/lease checks still apply.
            if self.phase == "policy" and now - self.last_observation > self.config["observation_timeout_s"]:
                raise TimeoutError("ARX policy observation expired")

    def _watchdog(self):
        period = 1.0 / self.config["control_hz"]
        while not self.stopped.wait(period):
            try:
                with self.lock:
                    now = time.monotonic()
                    self.check_health(now)
                    if (not self.single_arm and self.phase != "stopped"
                            and self.leader_mode == "mirror"):
                        desired = self.follower.fresh_state()["position"].copy()
                        desired[-1] = np.clip(desired[-1], 0, self.leader.robot.gripper_width)
                        self.leader.accept(desired, now - self.last_mirror, preview_s=period)
                    self.last_mirror = now
            except Exception as exc:
                with self.lock:
                    try:
                        self.stop(exc)
                    except Exception as stop_exc:
                        print(f"ARX stop error: {stop_exc}", flush=True)
                print(f"ARX watchdog stopped both arms: {exc}", flush=True)
                return

    def set_phase(self, phase):
        if phase not in {"idle", "policy", "human", "position"}:
            raise ValueError(f"Unknown ARX phase: {phase}")
        if phase != self.phase:
            # Clear old trajectories before changing gains. During positioning
            # BOTH arms are gravity compensated and can be moved independently.
            self.follower.mode = "transition"
            follower_mode = "drag" if (self.single_arm and phase in {"human", "position"}) else "hold"
            follower_state = self.follower.set_mode(follower_mode)
            self.filtered = None
            if phase == "policy":
                # Re-anchor both controller memories to the measured pose after
                # HIL/positioning.  The replay boundary uses [measured,
                # measured] outside policy, so the first policy target must use
                # the same anchor rather than an older accepted target.
                anchor = np.asarray(follower_state["position"], dtype=np.float64).copy()
                self.follower.last_target = anchor.copy()
                self.filtered = anchor.copy()
            self.gripper_pickup_width = (
                float(follower_state["position"][-1])
                if phase == "human" and not self.single_arm else None
            )
            self.gripper_pickup_previous = None
            self.phase = phase
            self.last_action = time.monotonic() - 1.0 / self.config["action_hz"]
            self.last_observation = time.monotonic()
        elif phase == "policy":
            # A normal quit can happen while the operator is waiting in the
            # pre-episode positioning prompt. Refresh the policy observation
            # lease before the final slow home return.
            self.last_observation = time.monotonic()
        self.set_leader_mode("drag" if phase in {"human", "position"} else "mirror")
        return {"phase": self.phase, "follower": self.state(self.follower), "leader": self.state(self.leader),
                "gripper_pickup_width_m": self.gripper_pickup_width}

    def set_leader_mode(self, mode):
        if mode not in {"drag", "mirror"}:
            raise ValueError("Leader mode must be drag or mirror")
        if self.single_arm:
            # ``mirror`` means policy/hold for the shared controller; ``drag``
            # is the only mode in which a human may physically move it.
            target_mode = "drag" if mode == "drag" else "hold"
            if self.follower.mode != target_mode:
                self.follower.set_mode(target_mode)
            self.leader_mode = mode
            self.last_mirror = time.monotonic()
            return {"mode": mode}
        if mode != self.leader_mode:
            self.leader.set_mode(mode)
            self.leader_mode = mode
            self.last_mirror = time.monotonic()
        return {"mode": mode}

    @staticmethod
    def state(arm):
        state = arm.fresh_state()
        return {"position": state["position"].tolist(), "timestamp": state["timestamp"],
                "feedback_age_s": state["feedback_age_s"], "emergency": state["emergency"], "mode": state["mode"],
                "gripper_readout_rad": state["gripper_readout_rad"]}

    def execution_profile(self):
        robot = self.follower.robot
        dt = min(1.0 / self.config["action_hz"], 0.1)
        speed = np.r_[np.minimum(robot.joint_vel_max, self.config["joint_speed"]),
                      min(robot.gripper_vel_max, self.config["gripper_speed"])]
        return {"contract": EXECUTION_CONTRACT, "step_dt_s": dt,
                "alpha": self.config["chunk_ema_alpha"] if self.config["chunk_filter"] == "ema" else 1.0,
                "max_step": (speed * dt).tolist(),
                "lower": np.r_[robot.joint_pos_min, 0.0].tolist(),
                "upper": np.r_[robot.joint_pos_max, robot.gripper_width].tolist()}

    def execution_state(self, measured=None):
        if measured is None:
            measured = self.follower.fresh_state()["position"]
        if self.phase != "policy":
            # Counterfactual policy handoff from a human observation resets
            # hold/filter to that observation's measured pose.
            return np.stack([measured, measured]).tolist()
        previous = measured if self.follower.last_target is None else self.follower.last_target
        filtered = measured if self.filtered is None else self.filtered
        return np.stack([previous, filtered]).tolist()

    def position_tick(self):
        if self.phase != "position":
            raise ValueError("ARX positioning samples require the position phase")
        # Keep the operator-loop deadline without submitting a movement target.
        result = {"follower": self.state(self.follower), "leader": self.state(self.leader)}
        self.last_action = time.monotonic()
        return result

    def return_home(self):
        """Return through the SDK home trajectory while policy leases are inactive."""
        self.phase = "idle"
        self.filtered = None
        self.last_observation = time.monotonic()
        self.last_action = time.monotonic()
        self.set_leader_mode("mirror")
        self.follower.return_home()
        if not self.single_arm:
            self.leader.return_home()
        self.fault = None
        return {"follower": self.state(self.follower),
                "leader": self.state(self.leader)}

    def target(self, value):
        if self.phase not in {"policy", "human"}:
            raise ValueError("ARX targets require an active policy/human phase; positioning is manual drag")
        if self.single_arm and self.phase == "human":
            raise ValueError("Single-arm HIL reads measured drag state; policy targets are disabled during human phase")
        desired = target_vector(value).copy()
        now = time.monotonic()
        pickup_completed = False
        if self.phase == "human" and self.gripper_pickup_width is not None:
            requested = float(np.clip(desired[-1], 0, self.follower.robot.gripper_width))
            gap = requested - self.gripper_pickup_width
            previous = self.gripper_pickup_previous
            tolerance = max(0.001, self.config["gripper_close_threshold"])
            if abs(gap) <= tolerance or (previous is not None and gap * (previous - self.gripper_pickup_width) < 0):
                self.gripper_pickup_width = None
                pickup_completed = True
                print("Human gripper pickup complete; follower now tracks leader width", flush=True)
            else:
                desired[-1] = self.gripper_pickup_width
            self.gripper_pickup_previous = requested
        if self.filtered is None:
            self.filtered = self.follower.fresh_state()["position"].copy()
        action_dt = now - self.last_action
        execution_before = None
        if self.phase == "policy":
            profile = self.execution_profile()
            execution_before = self.execution_state()
            previous, filtered = np.asarray(execution_before)
            desired = accepted_target(
                desired, previous, filtered, alpha=profile["alpha"],
                max_step=np.asarray(profile["max_step"]),
                lower=np.asarray(profile["lower"]), upper=np.asarray(profile["upper"]),
            )
            action_dt = profile["step_dt_s"]
        accepted, measured = self.follower.accept(
            desired, action_dt,
            preview_s=0.05,
            pre_limited=(self.phase == "policy"),
        )
        self.filtered = accepted.copy()
        self.last_action = now
        self.sequence += 1
        return {"sequence": self.sequence, "accepted": accepted.tolist(), "measured": measured.tolist(),
                "accepted_monotonic": now, "sdk_timestamp": float(self.follower.controller.get_timestamp()),
                "leader": self.state(self.leader), "gripper_pickup_pending": self.gripper_pickup_width is not None,
                "gripper_pickup_completed": pickup_completed, "execution_state_before": execution_before,
                "execution_state_after": self.execution_state(), "execution_dt_s": action_dt}

    def handle(self, message):
        if not isinstance(message, dict) or message.get("token") != self.token:
            return {"ok": False, "error": "Invalid ARX session owner"}
        command, data = message.get("cmd"), message.get("data")
        with self.lock:
            try:
                if command in {"STOP", "SHUTDOWN"}:
                    return {"ok": True, "data": self.stop()}
                if command == "RETURN_HOME":
                    return {"ok": True, "data": self.return_home()}
                if self.fault or self.phase == "stopped":
                    if command != "INFO":
                        raise RuntimeError(f"ARX stop is latched: {self.fault}")
                if command == "INFO":
                    result = {"profile": PROFILE, "follower": self.follower.description(), "leader": self.leader.description(),
                              "execution_transform": self.execution_profile(),
                              "camera": self.camera.metadata, "config": self.config, "fault": self.fault,
                              "stop_behavior": "damping; RETURN_HOME uses the SDK home trajectory"}
                elif command == "HEARTBEAT":
                    self.last_contact = time.monotonic()
                    result = {"phase": self.phase}
                else:
                    self.check_health()
                    if command == "OBSERVATION":
                        rgb, received, sequence, age = self.fresh_camera()
                        self.last_observation = time.monotonic()
                        result = {"follower": self.state(self.follower), "image_rgb": rgb.copy(),
                                  "image_received": received, "camera_sequence": sequence, "camera_age_s": age}
                        result["execution_state"] = self.execution_state(np.asarray(result["follower"]["position"]))
                    elif command == "LEADER":
                        # A shared single-arm controller has no target RPC
                        # during HIL.  Feedback reads are its keepalive while
                        # the operator is dragging, so do not trip the action
                        # timeout merely because no policy target is sent.
                        if self.single_arm and self.phase in {"human", "position"}:
                            self.last_action = time.monotonic()
                        result = self.state(self.leader)
                    elif command == "POSITION_TICK":
                        result = self.position_tick()
                    elif command == "PHASE":
                        result = self.set_phase(data)
                    elif command == "LEADER_MODE":
                        result = self.set_leader_mode("drag" if self.phase in {"human", "position"} else data)
                    elif command == "TARGET":
                        result = self.target(data)
                    else:
                        raise ValueError(f"Unsupported ARX command: {command}")
                return {"ok": True, "data": result}
            except Exception as exc:
                try:
                    self.stop(exc)
                except Exception as stop_exc:
                    return {"ok": False, "error": f"{exc}; {stop_exc}"}
                return {"ok": False, "error": str(exc)}

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        with self.lock:
            try:
                self.stop()
            finally:
                self._stop_preview()
                if self.camera is not None:
                    self.camera.close()
                arms = (self.follower,) if self.single_arm else (self.follower, self.leader)
                for arm in arms:
                    if arm is not None:
                        arm.controller = None
                for lock in self.interface_locks:
                    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not args.address.startswith("ipc:///tmp/rlt-arx-"):
        parser.error("ARX worker must use the private local IPC endpoint")
    closing = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: closing.set())
    worker = None
    try:
        with zmq.Context() as context, context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            # Claim the endpoint before any controller is constructed.
            socket.bind(args.address)
            worker = DualArmWorker(json.loads(args.config), args.token)
            ready_text = (
                "single arm holds its current measured pose"
                if worker.single_arm else "both arms hold their current measured pose"
            )
            print(f"RLT ARX Stage 2 worker ready; {ready_text}", flush=True)
            while not closing.is_set():
                worker.preview_tick()
                if not socket.poll(20):
                    continue
                message = socket.recv_pyobj()
                reply = worker.handle(message)
                socket.send_pyobj(reply)
                if reply.get("ok") and message.get("cmd") == "SHUTDOWN":
                    break
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    main()
