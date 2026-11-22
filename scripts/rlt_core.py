"""Shared runtime core for RL-Token-Pi05 online RL (used by both entry points).

Byte-identical components extracted from scripts/train_rlt_stage2_pi05.py and
scripts/eval_rlt_pi05_so101.py, kept here as the single source of truth:

- RerunVisualizationManager          background Rerun visualization thread
- TransitionSource/Transition/...    boundary snapshots and transition building
- Pi05ActionCache/LeaderSample/...   action cache and leader sampling
- ExecutionWindow/BoundaryContext/.. execution windows and takeover state
- quantile_normalize/_load_json/...  stats loading and quantile normalization
- parse_camera_map/parse_max_relative_target  config parsing

Any change here affects both entry scripts at once. If log attribution ever
needs to differ, pass a caller-provided logger instead of the module-level one.
"""
from __future__ import annotations

import enum
import json
import logging
import pickle
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# SO-101 order used by the legacy evaluator and by the training checkpoints.
# Keep this explicit because LeRobot feature dictionaries are not an ordering
# contract: a robot/dataset can legally expose the same names in another order.
SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARX_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7)) + ("gripper",)


def actor_contract_for_scale(scale: float) -> str:
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Actor residual scale must be finite and non-negative")
    return "vla_residual_v1" if scale > 0 else "paper_full_output_v1"


def checkpoint_actor_scale(checkpoint: dict) -> float:
    scale = float(checkpoint.get("runtime_provenance", {}).get("actor_residual_scale", 0.0))
    if checkpoint.get("actor_contract") != actor_contract_for_scale(scale):
        raise ValueError(f"Incompatible Actor contract {checkpoint.get('actor_contract')!r} for residual scale {scale}")
    return scale


def _canonical_joint_name(name: str) -> str:
    """Normalize LeRobot action feature names to bare SO-101 joint names."""
    value = str(name).strip()
    if value.startswith("action."):
        value = value[len("action."):]
    if value.endswith(".pos"):
        value = value[:-4]
    return value


def action_order_indices(source_names, target_names) -> tuple[int, ...]:
    """Return indices that reorder policy outputs into robot feature order.

    ``source_names`` describes the order used by the trained policy, while
    ``target_names`` is the order expected by the robot/dataset boundary.
    Duplicate, missing, or unknown joints fail closed before any hardware move.
    """
    source = tuple(_canonical_joint_name(name) for name in source_names)
    target = tuple(_canonical_joint_name(name) for name in target_names)
    allowed_orders = (set(SO101_JOINT_NAMES), set(ARX_JOINT_NAMES))
    if len(source) != len(target):
        raise ValueError(f"Action order length mismatch: source={len(source)} target={len(target)}")
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError(f"Action order contains duplicate joints: source={source}, target={target}")
    if set(source) not in allowed_orders:
        raise ValueError(f"Unsupported policy joint names: {source}")
    if set(target) != set(source):
        raise ValueError(f"Robot and policy joints differ: {source}, {target}")
    positions = {name: index for index, name in enumerate(source)}
    return tuple(positions[name] for name in target)


def reorder_action_tensor(action: torch.Tensor, *, source_names, target_names) -> torch.Tensor:
    """Reorder the final action dimension without changing batch/chunk axes."""
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"action must be a torch.Tensor, got {type(action).__name__}")
    indices = action_order_indices(source_names, target_names)
    if action.shape[-1] != len(indices):
        raise ValueError(
            f"Action tensor last dimension {action.shape[-1]} does not match action order {len(indices)}"
        )
    index = torch.tensor(indices, dtype=torch.long, device=action.device)
    return action.index_select(-1, index)

# ── Replay collection-phase ids (shared by Transition machinery) ──────────
COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


class RerunVisualizationManager:
    """Best-effort, non-blocking Rerun logging for live robot data."""

    def __init__(self, log_data, compress_images: bool, fps: float):
        self._log_data = log_data
        self._compress_images = compress_images
        self._period_s = 1.0 / fps
        self._condition = threading.Condition()
        self._observation: dict | None = None
        self._action: dict | None = None
        self._debug_observation: dict | None = None
        self._debug_action: dict | None = None
        self._running = False
        self._disabled = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _copy_values(values: dict) -> dict:
        copied = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, torch.Tensor):
                copied[key] = value.detach().cpu().numpy().copy()
            elif isinstance(value, (list, tuple)):
                copied[key] = np.asarray(value).copy()
            elif isinstance(value, (int, float, np.integer, np.floating)):
                copied[key] = value
        return copied

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="rerun-visualization", daemon=True)
        self._thread.start()

    def publish_observation(self, observation: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._observation = self._copy_values(observation)
            self._condition.notify()

    def publish_action(self, action: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._action = self._copy_values(action)
            self._condition.notify()

    def publish_debug(self, *, observation: dict | None = None, action: dict | None = None) -> None:
        """Queue lightweight diagnostic scalars and vectors for the live viewer."""
        if self._disabled:
            return
        with self._condition:
            if observation:
                if self._debug_observation is None:
                    self._debug_observation = {}
                self._debug_observation.update(self._copy_values(observation))
            if action:
                self._debug_action = self._copy_values(action)
            self._condition.notify()

    @staticmethod
    def _merge_values(primary: dict | None, debug: dict | None) -> dict | None:
        if primary is None and debug is None:
            return None
        merged = {} if primary is None else dict(primary)
        if debug:
            merged.update(debug)
        return merged

    def _run(self) -> None:
        next_log_time = 0.0
        while True:
            with self._condition:
                while (
                    self._running
                    and self._observation is None
                    and self._action is None
                    and self._debug_observation is None
                    and self._debug_action is None
                ):
                    self._condition.wait(timeout=0.1)
                if (
                    not self._running
                    and self._observation is None
                    and self._action is None
                    and self._debug_observation is None
                    and self._debug_action is None
                ):
                    return
                wait_s = next_log_time - time.monotonic()
                if self._running and wait_s > 0:
                    self._condition.wait(timeout=wait_s)
                    continue
                # Action receipts and their diagnostic comparison arrive from
                # adjacent calls in the control loop. Give the producer a short
                # coalescing window so one Rerun sample contains both.
                self._condition.wait(timeout=0.002)
                observation = self._merge_values(self._observation, self._debug_observation)
                action = self._merge_values(self._action, self._debug_action)
                self._observation = self._action = None
                self._debug_observation = self._debug_action = None
            try:
                self._log_data(
                    observation=observation,
                    action=action,
                    compress_images=self._compress_images,
                )
                next_log_time = time.monotonic() + self._period_s
            except Exception as exc:
                self._disabled = True
                logger.warning("Rerun visualization disabled after logging failure: %s", exc)
                return

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.5)


class RLTVisualizationManager(RerunVisualizationManager):
    """Stage-2 Rerun view with physical units, named series and an .rrd recording.

    The generic LeRobot logger is useful for arbitrary observation dictionaries,
    but it flattens the control contract into unnamed scalar arrays. Stage 2
    needs the VLA, Actor, SDK-accepted and measured signals aligned on one
    control timeline, so this manager publishes those entities explicitly.
    """

    _TARGET_COLORS = [(66, 133, 244, 255), (245, 124, 0, 255), (52, 168, 83, 255)]
    _ERROR_COLORS = [(156, 39, 176, 255), (229, 57, 53, 255)]

    def __init__(
        self,
        *,
        output_dir: Path,
        joint_names: tuple[str, ...] | list[str],
        compress_images: bool,
        fps: float,
        unit_system: str = "arx",
        session_name: str | None = None,
        spawn_viewer: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.joint_names = tuple(str(name) for name in joint_names)
        if unit_system not in {"arx", "so101"}:
            raise ValueError(f"Unsupported Rerun unit system: {unit_system!r}")
        self.unit_system = unit_system
        self.session_name = session_name
        self._spawn_viewer = bool(spawn_viewer)
        self._rr = None
        self._sequence = 0
        self._last_accepted: np.ndarray | None = None
        self._recording_path = self.output_dir / "stage2_visualization.rrd"
        super().__init__(self._log_sample, compress_images=compress_images, fps=fps)

    @property
    def recording_path(self) -> Path:
        return self._recording_path

    def start(self) -> None:
        import rerun as rr

        self.output_dir.mkdir(parents=True, exist_ok=True)
        app_id = (self.session_name or f"rlt_stage2_pi05_{self.output_dir.name}").replace("-", "_")
        rr.init(app_id, spawn=False)
        rr.save(self._recording_path)
        self._rr = rr
        self._send_blueprint()

        viewer_path = shutil.which("rerun") or shutil.which("rerun-cli")
        if self._spawn_viewer and viewer_path:
            try:
                rr.spawn(
                    executable_path=viewer_path,
                    hide_welcome_screen=True,
                    memory_limit="20%",
                    server_memory_limit="2GiB",
                )
                logger.info("Rerun Viewer connected using %s", viewer_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Rerun Viewer could not be started (%s). The recording will still be written to %s.",
                    exc, self._recording_path,
                )
        elif self._spawn_viewer:
            logger.warning(
                "No rerun viewer executable was found. The recording will be written to %s.",
                self._recording_path,
            )
        super().start()
        logger.info("Rerun recording: %s", self._recording_path)

    def stop(self) -> None:
        super().stop()
        if self._rr is not None:
            try:
                self._rr.rerun_shutdown()
            finally:
                self._rr = None

    def _path(self, group: str, joint: str, signal: str) -> str:
        return f"robot/{group}/{joint}/{signal}"

    def _send_blueprint(self) -> None:
        import rerun.blueprint as rrb

        arm = self.joint_names[:-1] if len(self.joint_names) > 1 else self.joint_names
        target_paths = [self._path("joints", name, "target") for name in arm]
        actual_paths = [self._path("joints", name, "actual") for name in arm]
        accepted_paths = [self._path("joints", name, "accepted") for name in arm]
        error_paths = [self._path("joints", name, "error") for name in arm]
        gripper_target = [self._path("gripper", "gripper", "target")]
        gripper_actual = [self._path("gripper", "gripper", "actual")]
        gripper_accepted = [self._path("gripper", "gripper", "accepted")]
        gripper_error = [self._path("gripper", "gripper", "error")]
        summary = [
            "robot/summary/source",
            "robot/summary/critical",
            "robot/summary/episode",
            "robot/summary/episode_step",
            "robot/summary/max_step_deg",
            "robot/summary/max_selected_vla_deg",
            "robot/summary/max_accepted_selected_deg",
        ]
        training = [
            "training/updates/total",
            "training/updates/critic",
            "training/updates/actor",
            "training/updates/aligned_critic",
            "training/q1_mean",
            "training/q2_mean",
            "training/td_error",
            "training/actor_q",
            "training/reference_q",
            "training/actor_advantage",
            "training/bc_loss",
            "training/shadow/mean_peak_step_deg",
            "training/shadow/actor_peak_step_deg",
            "training/shadow/vla_peak_step_deg",
            "training/shadow/mean_over_limit_steps",
            "training/shadow/actor_over_limit_steps",
            "training/shadow/vla_over_limit_steps",
            "training/shadow/peak_actor_vla_deg",
            "training/shadow/peak_actor_vla_gripper_mm",
        ]
        views = [
            rrb.Spatial2DView(origin="camera/wrist", name="Live camera"),
            rrb.TimeSeriesView(name="Joint targets (deg)", contents=target_paths),
            rrb.TimeSeriesView(name="Measured vs accepted (deg)", contents=actual_paths + accepted_paths),
            rrb.TimeSeriesView(name="Joint target error (deg)", contents=error_paths),
            rrb.TimeSeriesView(
                name=f"Gripper ({'mm' if self.unit_system == 'arx' else 'deg'})",
                contents=gripper_target + gripper_actual + gripper_accepted + gripper_error,
            ),
            rrb.TimeSeriesView(name="Robot state", contents=summary),
            rrb.TimeSeriesView(name="Training and shadow metrics", contents=training),
            rrb.TextLogView(name="Control events", contents=["status/events"]),
        ]
        self._rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views, grid_columns=2)))

        for name in arm:
            self._rr.log(
                self._path("joints", name, "target"),
                self._rr.SeriesLines(names=["VLA", "selected", "SDK accepted"], colors=self._TARGET_COLORS),
                static=True,
            )
            self._rr.log(
                self._path("joints", name, "error"),
                self._rr.SeriesLines(names=["selected - VLA", "accepted - selected"], colors=self._ERROR_COLORS),
                static=True,
            )
            self._rr.log(
                self._path("joints", name, "accepted"),
                self._rr.SeriesLines(names=["SDK accepted"], colors=[self._TARGET_COLORS[2]]),
                static=True,
            )
        self._rr.log(
            self._path("gripper", "gripper", "target"),
            self._rr.SeriesLines(names=["VLA", "selected", "SDK accepted"], colors=self._TARGET_COLORS),
            static=True,
        )
        self._rr.log(
            self._path("gripper", "gripper", "error"),
            self._rr.SeriesLines(names=["selected - VLA", "accepted - selected"], colors=self._ERROR_COLORS),
            static=True,
        )
        self._rr.log(
            self._path("gripper", "gripper", "accepted"),
            self._rr.SeriesLines(names=["SDK accepted"], colors=[self._TARGET_COLORS[2]]),
            static=True,
        )

    @staticmethod
    def _array(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        try:
            result = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return result if np.isfinite(result).all() else None

    @staticmethod
    def _image_array(value) -> np.ndarray | None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        try:
            result = np.asarray(value)
        except (TypeError, ValueError):
            return None
        if result.ndim not in (2, 3):
            return None
        if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
            return None
        return result

    def _find_array(self, values: dict, key: str) -> np.ndarray | None:
        for candidate in (key, f"action.{key}", f"observation.{key}"):
            if candidate in values:
                return self._array(values[candidate])
        return None

    def _joint_values(self, observation: dict) -> np.ndarray | None:
        values = []
        for name in self.joint_names:
            value = None
            for key in (f"{name}.pos", name):
                if key in observation:
                    value = self._array(observation[key])
                    break
            if value is None or value.size != 1:
                return None
            values.append(float(value.reshape(-1)[0]))
        return np.asarray(values, dtype=np.float64)

    def _camera(self, observation: dict) -> np.ndarray | None:
        preferred = ("cam_wrist", "observation.images.cam_wrist", "wrist")
        for key in preferred:
            if key in observation:
                value = self._image_array(observation[key])
                if value is not None:
                    return value
        for key, value in observation.items():
            if str(key).endswith(".pos") or str(key).startswith("_"):
                continue
            array = self._image_array(value)
            if array is not None and min(array.shape[-2:]) > 16:
                return array
        return None

    def _physical(self, values: np.ndarray) -> np.ndarray:
        result = values.astype(np.float64, copy=True)
        if result.size and self.unit_system == "arx":
            result[:-1] = np.rad2deg(result[:-1])
            result[-1] *= 1000.0
        return result

    def _log_scalar(self, path: str, value: float) -> None:
        if np.isfinite(value):
            self._rr.log(path, self._rr.Scalars(float(value)))

    def _log_vector_series(self, path: str, values: np.ndarray, names: list[str]) -> None:
        if values.size == len(names) and np.isfinite(values).all():
            self._rr.log(path, self._rr.Scalars(values.astype(np.float64)))

    def _log_sample(self, observation: dict | None, action: dict | None, compress_images: bool = False) -> None:
        if self._rr is None:
            return
        self._rr.set_time("control", sequence=self._sequence)
        self._sequence += 1

        if observation:
            camera = self._camera(observation)
            if camera is not None:
                if camera.ndim == 3 and camera.shape[0] in (1, 3, 4) and camera.shape[-1] not in (1, 3, 4):
                    camera = np.transpose(camera, (1, 2, 0))
                if camera.ndim == 2:
                    camera = np.repeat(camera[..., None], 3, axis=-1)
                if camera.dtype != np.uint8:
                    camera = np.clip(camera, 0, 255).astype(np.uint8)
                entity = self._rr.Image(camera)
                if compress_images:
                    entity = entity.compress()
                self._rr.log("camera/wrist", entity)

            measured = self._joint_values(observation)
            if measured is not None:
                measured_physical = self._physical(measured)
                for index, name in enumerate(self.joint_names[:-1]):
                    self._log_scalar(self._path("joints", name, "actual"), measured_physical[index])
                if self.joint_names:
                    self._log_scalar(self._path("gripper", "gripper", "actual"), measured_physical[-1])

        if not action:
            return
        vla = self._find_array(action, "rlt/vla_target")
        selected = self._find_array(action, "rlt/selected_target")
        accepted = self._find_array(action, "rlt/accepted_target")
        if accepted is None:
            accepted = self._joint_values(action)
        if vla is not None and vla.size == len(self.joint_names):
            vla = vla.reshape(-1)
        if selected is not None and selected.size == len(self.joint_names):
            selected = selected.reshape(-1)
        if accepted is not None and accepted.size == len(self.joint_names):
            accepted = accepted.reshape(-1)
        if vla is not None and selected is not None and accepted is not None:
            vla_physical = self._physical(vla)
            selected_physical = self._physical(selected)
            accepted_physical = self._physical(accepted)
            for index, name in enumerate(self.joint_names[:-1]):
                self._log_vector_series(
                    self._path("joints", name, "target"),
                    np.asarray([vla_physical[index], selected_physical[index], accepted_physical[index]]),
                    ["VLA", "selected", "SDK accepted"],
                )
                self._log_vector_series(
                    self._path("joints", name, "error"),
                    np.asarray([
                        selected_physical[index] - vla_physical[index],
                        accepted_physical[index] - selected_physical[index],
                    ]),
                    ["selected - VLA", "accepted - selected"],
                )
                self._log_scalar(self._path("joints", name, "accepted"), accepted_physical[index])
            self._log_vector_series(
                self._path("gripper", "gripper", "target"),
                np.asarray([vla_physical[-1], selected_physical[-1], accepted_physical[-1]]),
                ["VLA", "selected", "SDK accepted"],
            )
            self._log_vector_series(
                self._path("gripper", "gripper", "error"),
                np.asarray([
                    selected_physical[-1] - vla_physical[-1],
                    accepted_physical[-1] - selected_physical[-1],
                ]),
                ["selected - VLA", "accepted - selected"],
            )
            self._log_scalar(self._path("gripper", "gripper", "accepted"), accepted_physical[-1])
            if self._last_accepted is not None:
                step = np.abs(accepted_physical[:-1] - self._last_accepted[:-1])
                self._log_scalar("robot/summary/max_step_deg", float(step.max()))
            self._last_accepted = accepted_physical
            self._log_scalar("robot/summary/max_selected_vla_deg", float(np.abs(selected_physical[:-1] - vla_physical[:-1]).max()))
            self._log_scalar("robot/summary/max_accepted_selected_deg", float(np.abs(accepted_physical[:-1] - selected_physical[:-1]).max()))

        source = self._find_array(action, "rlt/source")
        critical = self._find_array(action, "rlt/critical")
        episode = self._find_array(action, "rlt/episode")
        episode_step = self._find_array(action, "rlt/episode_step")
        for path, value in (
            ("robot/summary/source", source),
            ("robot/summary/critical", critical),
            ("robot/summary/episode", episode),
            ("robot/summary/episode_step", episode_step),
        ):
            if value is not None and value.size == 1:
                self._log_scalar(path, float(value.reshape(-1)[0]))
        actor_advantage = self._find_array(action, "rlt/actor_advantage")
        if actor_advantage is not None and actor_advantage.size == 1:
            self._log_scalar("training/actor_advantage", float(actor_advantage.reshape(-1)[0]))

        source_value = int(source.reshape(-1)[0]) if source is not None and source.size == 1 else 0
        source_name = "Actor" if source_value == 1 else "VLA"
        critical_value = int(critical.reshape(-1)[0]) if critical is not None and critical.size == 1 else 0
        step_value = int(episode_step.reshape(-1)[0]) if episode_step is not None and episode_step.size == 1 else -1
        self._rr.log(
            "status/events",
            self._rr.TextLog(f"source={source_name} critical={critical_value} control_step={step_value}"),
        )

        if observation:
            for key, value in observation.items():
                if not str(key).startswith("observation.rlt/"):
                    continue
                array = self._array(value)
                if array is None or array.size != 1:
                    continue
                metric = str(key).removeprefix("observation.rlt/").replace("/", "_")
                self._log_scalar(f"training/{metric}", float(array.reshape(-1)[0]))


class TransitionSource(enum.IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


def collection_phase_to_id(phase: str) -> int:
    name = str(phase).split(":", 1)[0].lower()
    if name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def _ensure_array(value, *, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


@dataclass
class Transition:
    z_rl: np.ndarray           # (rlt_hidden_dim,)
    proprio: np.ndarray        # (state_dim,)
    ref_chunk: np.ndarray      # (n_action_steps_rl, action_dim)
    action_chunk: np.ndarray   # (n_action_steps_rl, action_dim)
    rewards: np.ndarray        # (n_action_steps_rl,)
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source: int
    source_chunk: np.ndarray   # uint8 (n_action_steps_rl,)
    valid_mask: np.ndarray     # bool (n_action_steps_rl,)
    executed_steps: int
    collection_phase: str
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0
    requested_chunk: np.ndarray | None = None  # pre-SDK requested targets, normalized
    execution_state: np.ndarray | None = None  # physical (previous, filtered) targets
    next_execution_state: np.ndarray | None = None
    # Model reference stays C rows; accepted targets/rewards use command ticks.
    reference_targets: np.ndarray | None = None
    preparation: dict[str, np.ndarray] | None = None
    next_preparation: dict[str, np.ndarray] | None = None
    execution_timing: np.ndarray | None = None
    vla_ref_chunk: np.ndarray | None = None  # Original VLA plan before intervention targets replace any rows.

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "z_rl": _ensure_array(self.z_rl, dtype=np.float16),
            "proprio": _ensure_array(self.proprio, dtype=np.float32),
            "ref_chunk": _ensure_array(self.ref_chunk, dtype=np.float16),
            "vla_ref_chunk": _ensure_array(self.ref_chunk if self.vla_ref_chunk is None else self.vla_ref_chunk,
                                           dtype=np.float16),
            "action_chunk": _ensure_array(self.action_chunk, dtype=np.float16),
            "requested_chunk": _ensure_array(
                self.action_chunk if self.requested_chunk is None else self.requested_chunk,
                dtype=np.float16,
            ),
            "rewards": _ensure_array(self.rewards, dtype=np.float32),
            "done": _ensure_array(self.done, dtype=np.bool_),
            "next_z_rl": _ensure_array(self.next_z_rl, dtype=np.float16),
            "next_proprio": _ensure_array(self.next_proprio, dtype=np.float32),
            "next_ref_chunk": _ensure_array(self.next_ref_chunk, dtype=np.float16),
            "source": _ensure_array(self.source, dtype=np.uint8),
            "source_chunk": _ensure_array(self.source_chunk, dtype=np.uint8),
            "valid_mask": _ensure_array(self.valid_mask, dtype=np.bool_),
            "executed_steps": _ensure_array(self.executed_steps, dtype=np.int16),
            "collection_phase_id": _ensure_array(
                collection_phase_to_id(self.collection_phase), dtype=np.uint8,
            ),
            "success": _ensure_array(self.success, dtype=np.int8),
            "intervention_flag": _ensure_array(self.intervention_flag, dtype=np.bool_),
            "episode_id": _ensure_array(self.episode_id, dtype=np.int32),
            "step_id": _ensure_array(self.step_id, dtype=np.int32),
            "execution_state": _ensure_array(
                np.zeros((2, self.action_chunk.shape[-1])) if self.execution_state is None else self.execution_state,
                dtype=np.float32,
            ),
            "next_execution_state": _ensure_array(
                np.zeros((2, self.action_chunk.shape[-1])) if self.next_execution_state is None else self.next_execution_state,
                dtype=np.float32,
            ),
            "execution_valid": _ensure_array(self.execution_state is not None, dtype=np.bool_),
            "next_execution_valid": _ensure_array(self.next_execution_state is not None, dtype=np.bool_),
            "reference_targets": _ensure_array(
                self.ref_chunk if self.reference_targets is None else self.reference_targets, dtype=np.float32,
            ),
            "execution_timing": _ensure_array(
                np.zeros(7) if self.execution_timing is None else self.execution_timing, dtype=np.float32,
            ),
            **self._preparation_arrays(self.preparation, "prepare_"),
            **self._preparation_arrays(self.next_preparation, "next_prepare_"),
        }

    def _preparation_arrays(self, preparation, prefix):
        dimension = self.action_chunk.shape[-1]
        values = preparation if preparation is not None else {
            "context": np.zeros((0, dimension)), "context_valid": np.zeros(0), "phase": 0.0,
            "blend_anchor": np.zeros(dimension), "blend_weights": np.ones(len(self.action_chunk)),
        }
        if "handoff" not in values:
            values = dict(values, handoff=np.zeros((2, dimension)), previous_policy=-1)
        return {prefix + key: _ensure_array(value, dtype=np.float32) for key, value in values.items()}


@dataclass
class Pi05ActionCache:
    """Full Pi0.5 raw chunks plus the model-space actions used by RLT."""

    raw_actions: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    z_rl: torch.Tensor | None = None
    next_index: int = 0

    def exhausted(self) -> bool:
        return self.raw_actions is None or self.next_index >= self.raw_actions.shape[1]

    def remaining_raw_actions(self) -> torch.Tensor | None:
        if self.exhausted() or self.raw_actions is None:
            return None
        return self.raw_actions[:, self.next_index:].clone()

    def refresh(self, raw_actions: torch.Tensor, z_rl: torch.Tensor, action_dim: int) -> None:
        self.raw_actions = raw_actions
        self.actions = raw_actions[:, :, :action_dim]
        self.z_rl = z_rl
        self.next_index = 0

    def clear(self) -> None:
        self.raw_actions = None
        self.actions = None
        self.z_rl = None
        self.next_index = 0

    def take(self, count: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if self.exhausted() or self.actions is None or self.z_rl is None:
            raise RuntimeError("PI0.5 action cache is empty.")
        if count < 1 or self.next_index + count > self.actions.shape[1]:
            raise ValueError("Requested Pi0.5 action horizon is not available in the cache.")
        action_index = self.next_index
        result = self.actions[:, action_index : action_index + count]
        self.next_index += count
        return result, action_index, self.z_rl


@dataclass(frozen=True)
class LeaderSample:
    sequence: int
    timestamp: float
    action: dict[str, float]
    tensor: torch.Tensor
    healthy: bool = True

    def is_fresh(self, max_age_s: float, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return self.healthy and current - self.timestamp <= max_age_s


@dataclass
class BoundaryContext:
    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor
    collection_phase: str
    episode_id: int
    step_id: int
    execution_state: np.ndarray | None = None
    # Physical follower state at this boundary (ARX rad/m). ``proprio`` is
    # quantile-normalized and must never be mixed with plan targets.
    measured_state: torch.Tensor | None = None
    preparation: dict[str, torch.Tensor] | None = None
    reference_targets: torch.Tensor | None = None
    execution_timing: np.ndarray | None = None


@dataclass
class ExecutionWindow:
    """Actions actually accepted by hardware under one observation boundary."""

    boundary: BoundaryContext
    actions: list[torch.Tensor] = field(default_factory=list)
    requested_actions: list[torch.Tensor | None] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    intervention_flags: list[bool] = field(default_factory=list)

    def append(
        self,
        actual_action: torch.Tensor,
        source: int,
        intervention: bool = False,
        *,
        requested_action: torch.Tensor | None = None,
    ) -> None:
        if actual_action.ndim != 1:
            raise ValueError(f"Executed action must be rank 1, got {tuple(actual_action.shape)}")
        self.actions.append(actual_action.detach().clone())
        self.requested_actions.append(
            None if requested_action is None else requested_action.detach().clone()
        )
        self.sources.append(int(source))
        self.intervention_flags.append(bool(intervention))

    @property
    def executed_steps(self) -> int:
        return len(self.actions)


def _resolve_chunk_source(sources: list[int], intervention_flags: list[bool]) -> tuple[int, bool]:
    intervention = any(intervention_flags)
    source_set = set(sources)
    has_human = int(TransitionSource.HUMAN) in source_set
    has_policy = any(s in source_set for s in (
        int(TransitionSource.BASE), int(TransitionSource.RL), int(TransitionSource.MIXED),
    ))
    if int(TransitionSource.MIXED) in source_set or (has_human and has_policy):
        return int(TransitionSource.MIXED), intervention
    if has_human or intervention:
        return int(TransitionSource.HUMAN), intervention
    return int(sources[0]), intervention


def quantile_normalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return 2.0 * (tensor - q01) / denom - 1.0


def _load_json(path: Path) -> dict:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_quantiles(stats: dict, dimension: int = 6) -> dict[str, list[float]]:
    result = {}
    for key in ("observation.state", "action"):
        if key not in stats or not isinstance(stats[key], dict):
            raise ValueError(f"Normalization stats missing {key!r}")
        q01 = np.asarray(stats[key].get("q01"), dtype=np.float32)
        q99 = np.asarray(stats[key].get("q99"), dtype=np.float32)
        if q01.shape != (dimension,) or q99.shape != (dimension,):
            raise ValueError(f"{key} quantiles must each be exactly {dimension}D")
        if not np.isfinite(q01).all() or not np.isfinite(q99).all():
            raise ValueError(f"{key} quantiles must be finite")
        if not np.all(q99 > q01):
            raise ValueError(f"{key} requires q99 > q01 elementwise")
        result[key] = {"q01": q01.tolist(), "q99": q99.tolist()}
    return result


def load_normalization_stats(pi05_path: Path, explicit_stats_path: str | None) -> tuple[dict, str]:
    config = _load_json(pi05_path / "config.json")
    dimension = config["output_features"]["action"]["shape"][0]
    if dimension not in (6, 7) or config["input_features"]["observation.state"]["shape"] != [dimension]:
        raise ValueError("Expected matching SO-101 6D or ARX 7D state/action features")
    if config.get("use_relative_actions", False):
        raise ValueError("Stage 2 requires absolute actions")
    if explicit_stats_path:
        path = Path(explicit_stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit stats path does not exist: {path}")
        raw = _load_json(path)
        if raw.get("use_relative_actions") is True:
            raise ValueError("Stage 2 requires absolute actions; use_relative_actions must be false")
        return _validate_quantiles(raw, dimension), str(path.resolve())

    processor_path = pi05_path / "policy_preprocessor.json"
    if not processor_path.is_file():
        raise FileNotFoundError("Checkpoint processor state is unavailable; pass --stats_path explicitly")
    processor = _load_json(processor_path)
    normalizers = [
        step for step in processor.get("steps", [])
        if step.get("registry_name") == "normalizer_processor"
    ]
    if len(normalizers) != 1:
        raise ValueError("Checkpoint must declare exactly one normalizer_processor")
    normalizer_config = normalizers[0].get("config", {})
    norm_map = normalizer_config.get("norm_map", {})
    if norm_map.get("STATE") != "QUANTILES" or norm_map.get("ACTION") != "QUANTILES":
        raise ValueError("Checkpoint STATE and ACTION normalization must both be QUANTILES")
    if normalizer_config.get("use_relative_actions") is True:
        raise ValueError("Checkpoint use_relative_actions must be false")

    candidates = sorted(pi05_path.glob("*normalizer*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            "Checkpoint declares normalization but has no normalizer safetensors; pass --stats_path explicitly"
        )
    from safetensors.torch import load_file
    tensors = {}
    for candidate in candidates:
        tensors.update(load_file(str(candidate)))
    aliases = {
        "observation.state": ("observation.state", "state"),
        "action": ("action",),
    }
    extracted = {}
    for feature, prefixes in aliases.items():
        for prefix in prefixes:
            q01 = next((v for k, v in tensors.items() if k in (f"{prefix}.q01", f"{prefix}/q01")), None)
            q99 = next((v for k, v in tensors.items() if k in (f"{prefix}.q99", f"{prefix}/q99")), None)
            if q01 is not None and q99 is not None:
                extracted[feature] = {"q01": q01.cpu().tolist(), "q99": q99.cpu().tolist()}
                break
    return _validate_quantiles(extracted, dimension), ",".join(str(p.resolve()) for p in candidates)


def parse_camera_map(
    camera_map_json: str,
    required_features: list[str],
    configured_cameras: set[str],
    pad_missing: bool = False,
) -> dict[str, str]:
    if not camera_map_json:
        raise ValueError(
            "--camera_map is required. Map every physical camera to every image feature required by the checkpoint."
        )
    try:
        mapping = json.loads(camera_map_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--camera_map must be valid JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("--camera_map must be a JSON object of physical camera names to checkpoint feature names.")
    physical_names = set(mapping)
    targets = list(mapping.values())
    if physical_names != configured_cameras:
        raise ValueError(
            f"Camera map keys {sorted(physical_names)} must exactly match configured cameras {sorted(configured_cameras)}."
        )
    if len(set(targets)) != len(targets):
        raise ValueError("Each physical camera must map to a distinct checkpoint image feature.")
    unknown = sorted(set(targets) - set(required_features))
    if unknown:
        raise ValueError(f"Camera map targets unknown checkpoint features: {unknown}")
    mapped_set = set(targets)
    missing_set = set(required_features) - mapped_set
    if missing_set:
        if not pad_missing:
            missing = sorted(missing_set)
            raise ValueError(
                f"Camera mapping does not satisfy checkpoint contract. Missing={missing}, "
                f"checkpoint_features={required_features}. "
                f"Supply the missing cameras or pass --pad_missing_cameras for test-only runs."
            )
        logger.warning(
            "TEST-ONLY MODE: unmapped camera features %s will be filled with zero-pixel tensors.",
            sorted(missing_set),
        )
    return mapping


def parse_max_relative_target(value: str | None) -> float | dict[str, float] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError("--max_relative_target must be a positive scalar or a JSON object.") from exc
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError("--max_relative_target scalar must be finite and positive.")
        return float(parsed)
    if isinstance(parsed, dict) and parsed and all(
        isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
        and np.isfinite(v) and v > 0 for k, v in parsed.items()
    ):
        return {key: float(limit) for key, limit in parsed.items()}
    raise ValueError("--max_relative_target JSON must be a non-empty map of motor names to positive finite limits.")


__all__ = [
    "actor_contract_for_scale", "checkpoint_actor_scale",
    "SO101_JOINT_NAMES",
    "action_order_indices",
    "reorder_action_tensor",
    "RerunVisualizationManager",
    "RLTVisualizationManager",
    "TransitionSource",
    "collection_phase_to_id",
    "_ensure_array",
    "Transition",
    "Pi05ActionCache",
    "LeaderSample",
    "BoundaryContext",
    "ExecutionWindow",
    "_resolve_chunk_source",
    "quantile_normalize",
    "_load_json",
    "_validate_quantiles",
    "load_normalization_stats",
    "parse_camera_map",
    "parse_max_relative_target",
]


def write_warmup_interval_records(
    out_file,
    ep: int,
    frames_in_interval: list,
    frame_data: dict,
    *,
    horizon: int,
    stride: int,
    action_dim: int,
) -> int:
    """Build stride-2 replay windows inside one annotated interval and write warmup
    journal records (pickle stream, schema identical to Stage-2 online transitions
    with ``collection_phase_id=WARMUP``). Shared by ``extract_annotated_warmup.py``
    and ``extract_warmup_from_cache.py``.

    ``frame_data`` maps ``(episode, frame) -> (z_rl fp16, proprio fp32, action fp32)``.
    Returns the number of records written.
    """
    f = out_file  # verbatim body below references ``f``
    n_records = 0
    for pos in range(0, len(frames_in_interval), stride):
        window_frames = frames_in_interval[pos:pos + horizon]
        if len(window_frames) < 1:
            break
        t0f = window_frames[0]
        t1f = frames_in_interval[min(pos + horizon, len(frames_in_interval) - 1)]
        z0, p0, _ = frame_data[(ep, t0f)]
        z1, p1, _ = frame_data[(ep, t1f)]

        action_chunk = np.zeros((horizon, action_dim), dtype=np.float32)
        next_ref_chunk = np.zeros((horizon, action_dim), dtype=np.float32)
        valid_mask = np.zeros(horizon, dtype=np.bool_)
        executed = 0
        for i, fr in enumerate(window_frames):
            action_chunk[i] = frame_data[(ep, fr)][2]
            valid_mask[i] = True
            executed += 1
        # next window's demo actions as next_ref_chunk (may be all-zero padding)
        next_frames = frames_in_interval[pos + horizon:pos + 2 * horizon]
        for i, fr in enumerate(next_frames):
            next_ref_chunk[i] = frame_data[(ep, fr)][2]

        record = {
            "z_rl": z0.astype(np.float16),
            "proprio": p0.astype(np.float32),
            "ref_chunk": action_chunk.astype(np.float16),
            "action_chunk": action_chunk.astype(np.float16),
            "rewards": np.zeros(horizon, dtype=np.float32),
            "done": np.asarray(False, dtype=np.bool_),
            "next_z_rl": z1.astype(np.float16),
            "next_proprio": p1.astype(np.float32),
            "next_ref_chunk": next_ref_chunk.astype(np.float16),
            "source": np.asarray(TransitionSource.BASE, dtype=np.uint8),
            "source_chunk": np.full(horizon, TransitionSource.BASE, dtype=np.uint8),
            "valid_mask": valid_mask,
            "executed_steps": np.asarray(executed, dtype=np.int16),
            "collection_phase_id": np.asarray(COLLECTION_PHASE_WARMUP, dtype=np.uint8),
            "success": np.asarray(0, dtype=np.int8),
            "intervention_flag": np.asarray(False, dtype=np.bool_),
            "episode_id": np.asarray(ep, dtype=np.int32),
            "step_id": np.asarray(t0f, dtype=np.int32),
        }
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_records += 1
    return n_records
