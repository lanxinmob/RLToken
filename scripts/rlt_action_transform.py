"""ARX policy target contract shared by the SDK worker and online learner.

No SDK or torch import at module load: the hardware Python only needs numpy.
Keep every annotation Python 3.9-compatible (the ARX worker interpreter):
no ``X | None`` unions, no ``from __future__`` tricks.
Policy slew budgets are fixed per submitted action, not accumulated during
inference waits. Human/positioning control keeps its separate SDK path.
"""

import numpy as np

EXECUTION_CONTRACT = "arx_accepted_target_v1"
EXECUTION_STATE_KEY = "_rlt_execution_state"


def accepted_target(requested, previous, filtered, *, alpha, max_step, lower, upper):
    """One physical-unit policy step; numpy and torch use identical algebra."""
    if isinstance(requested, np.ndarray):
        minimum, maximum = np.minimum, np.maximum
    else:
        import torch

        minimum, maximum = torch.minimum, torch.maximum
    desired = filtered + alpha * (requested - filtered)
    delta = minimum(maximum(desired - previous, -max_step), max_step)
    return minimum(maximum(previous + delta, lower), upper)


def smoothing_kernel(taps: int):
    """Zero-phase triangular FIR; ``taps <= 1`` disables smoothing.

    The 5-tap kernel [1, 2, 3, 2, 1]/9 has an exact null at ``fs/3`` (10 Hz at
    30 Hz control), which is where single-chunk sampling noise lives, while a
    1-3 Hz trajectory passes almost untouched.
    """
    if taps <= 1:
        return None
    if taps % 2 == 0:
        raise ValueError("chunk smooth taps must be odd (0 disables smoothing)")
    half = taps // 2
    weights = np.array([half + 1 - abs(offset) for offset in range(-half, half + 1)], dtype=np.float64)
    return weights / weights.sum()


def smooth_actions_numpy(actions, taps: int, *, joints=None):
    """Filter a ``(time, action_dim)`` chunk along time; the last column is left alone."""
    kernel = smoothing_kernel(taps)
    if kernel is None:
        return actions
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"smooth_actions_numpy expects (time, action_dim), got {array.shape}")
    joints = array.shape[1] - 1 if joints is None else min(int(joints), array.shape[1])
    pad = len(kernel) // 2
    result = array.copy()
    padded = np.pad(array[:, :joints], ((pad, pad), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, len(kernel), axis=0)
    result[:, :joints] = (windows * kernel).sum(axis=-1)
    return result


def smooth_actions_torch(actions, taps: int, *, joints=None):
    """Torch twin of :func:`smooth_actions_numpy`; differentiable for the actor Q path."""
    kernel = smoothing_kernel(taps)
    if kernel is None:
        return actions
    import torch
    import torch.nn.functional as functional

    if actions.ndim != 3:
        raise ValueError(f"smooth_actions_torch expects (batch, time, action_dim), got {tuple(actions.shape)}")
    dimension = actions.shape[-1]
    joints = dimension - 1 if joints is None else min(int(joints), dimension)
    pad = len(kernel) // 2
    weights = torch.as_tensor(kernel, dtype=actions.dtype, device=actions.device)
    block = actions[..., :joints].transpose(1, 2)  # (B, J, T)
    block = functional.pad(block, (pad, pad), mode="replicate")
    windows = block.unfold(-1, len(kernel), 1)  # (B, J, T, K)
    smoothed = (windows * weights).sum(dim=-1).transpose(1, 2)
    if joints == dimension:
        return smoothed
    return torch.cat([smoothed, actions[..., joints:]], dim=-1)


class ActionTimeline:
    """Model waypoints -> command window, shared by rollout, BC and Q.

    The grid controls interpolation density; tick_hz controls physical time.
    Context contains the frozen VLA flanks around the actor's model window.
    Keeping those flanks and the fractional grid phase makes slicing a full
    smoothed plan identical to preparing an actor replacement of that slice.
    """

    def __init__(self, model_steps, action_hz, grid_hz, tick_hz, smooth_taps=0, start_blend_steps=0):
        if not all(np.isfinite(v) and v > 0 for v in (action_hz, grid_hz, tick_hz)):
            raise ValueError("Action, grid and command rates must be finite and positive")
        if grid_hz < action_hz:
            raise ValueError("Execution grid must be at least the model action rate")
        smoothing_kernel(smooth_taps)
        self.model_steps = int(model_steps)
        self.action_hz, self.grid_hz, self.tick_hz = map(float, (action_hz, grid_hz, tick_hz))
        self.smooth_taps = int(smooth_taps)
        self.start_blend_steps = int(start_blend_steps)
        self.steps = int(np.floor((self.model_steps - 1) * self.grid_hz / self.action_hz + 1e-9)) + 1
        # One additional point is needed to interpolate a fractional final row.
        self.pad = self.smooth_taps // 2 + 1 if self.smooth_taps > 1 else 1
        self.signature = np.asarray([
            1, self.model_steps, self.action_hz, self.grid_hz, self.tick_hz,
            self.smooth_taps, self.start_blend_steps,
        ], dtype=np.float32)

    def profile(self):
        return dict(zip(
            ("version", "model_steps", "action_hz", "grid_hz", "tick_hz", "smooth_taps", "start_blend_steps"),
            self.signature.tolist(),
        ))

    def context(self, full_actions, model_start):
        import torch

        indices = torch.cat([
            torch.arange(model_start - self.pad, model_start, device=full_actions.device),
            torch.arange(model_start + self.model_steps, model_start + self.model_steps + self.pad,
                         device=full_actions.device),
        ]).clamp(0, full_actions.shape[1] - 1)
        return full_actions.index_select(1, indices)

    def metadata(self, full_actions, model_start, *, phase=0.0, step_id=0, blend_anchor=None):
        import torch

        batch, _, dimension = full_actions.shape
        options = {"dtype": full_actions.dtype, "device": full_actions.device}
        weights = torch.ones(batch, self.steps, **options)
        if self.start_blend_steps:
            weights = ((torch.arange(self.steps, **options) + step_id + 1) / self.start_blend_steps).clamp(max=1)
            weights = weights.unsqueeze(0).expand(batch, -1)
        if blend_anchor is None:
            if (weights < 1).any():
                raise ValueError("Startup blend requires the measured physical pose normalized as an action")
            blend_anchor = torch.zeros(batch, dimension, **options)
        indices = torch.cat([
            torch.arange(model_start - self.pad, model_start, device=full_actions.device),
            torch.arange(model_start + self.model_steps, model_start + self.model_steps + self.pad,
                         device=full_actions.device),
        ])
        return {
            "context": self.context(full_actions, model_start),
            # A padded full-plan endpoint depends on the Actor only when the
            # replacement includes that endpoint. For a slice starting at 1,
            # negative FIR indices still replicate the frozen VLA action 0.
            "context_valid": (~(
                ((indices < 0) & (model_start == 0))
                | ((indices >= full_actions.shape[1])
                   & (model_start + self.model_steps >= full_actions.shape[1]))
            )).to(full_actions).expand(batch, -1),
            "phase": torch.full((batch,), float(phase), **options),
            "blend_anchor": torch.as_tensor(blend_anchor, **options).reshape(batch, dimension),
            "blend_weights": weights,
        }

    def prepare(self, actions, metadata, *, execution_scale=1.0):
        """Differentiable normalized pre-SDK targets, including startup blending."""
        import torch

        context = metadata["context"].to(actions)
        if context.shape != (actions.shape[0], 2 * self.pad, actions.shape[-1]):
            raise ValueError("Invalid full-plan smoothing context")
        actions = actions * execution_scale
        edge = torch.cat([actions[:, :1].expand(-1, self.pad, -1),
                          actions[:, -1:].expand(-1, self.pad, -1)], dim=1)
        # At the actual full-plan ends, FIR edge replication belongs to the
        # replacement Actor endpoint too; it must not retain the old VLA value.
        context = torch.where(metadata["context_valid"].to(actions).unsqueeze(-1) > .5, context, edge)
        full = torch.cat([context[:, :self.pad], actions, context[:, self.pad:]], dim=1)
        full = smooth_actions_torch(full, self.smooth_taps)
        coordinates = (
            torch.arange(self.steps, device=actions.device, dtype=actions.dtype)[None]
            * (self.action_hz / self.grid_hz)
            + metadata["phase"].to(actions).reshape(-1, 1) + self.pad
        )
        left = coordinates.floor().long().clamp(0, full.shape[1] - 1)
        right = (left + 1).clamp_max(full.shape[1] - 1)
        fraction = (coordinates - left).unsqueeze(-1)
        shape = (-1, -1, actions.shape[-1])
        result = full.gather(1, left.unsqueeze(-1).expand(*shape)) * (1 - fraction)
        result = result + full.gather(1, right.unsqueeze(-1).expand(*shape)) * fraction
        weights = metadata["blend_weights"].to(actions).unsqueeze(-1)
        anchor = metadata["blend_anchor"].to(actions).unsqueeze(1)
        # Keep the gripper unblended, matching Stage 2 vla-only execution.
        return torch.cat([anchor[..., :-1] + weights * (result[..., :-1] - anchor[..., :-1]),
                          result[..., -1:]], dim=-1)


class AcceptedActionTransform:
    """Differentiable normalized chunk -> normalized SDK accepted targets."""

    def __init__(self, profile, action_q01, action_q99, *, device="cpu", execution_scale=1.0,
                 smooth_taps: int = 0, timeline=None):
        import torch

        if profile["contract"] != EXECUTION_CONTRACT:
            raise ValueError(f"Unsupported execution transform: {profile['contract']}")
        self.profile = profile
        self.timeline = timeline
        self.smooth_taps = int(smooth_taps)
        if self.smooth_taps > 1 and self.smooth_taps % 2 == 0:
            raise ValueError("chunk smooth taps must be odd (0 disables smoothing)")
        self.execution_scale = float(execution_scale)
        self.q01 = torch.as_tensor(action_q01, dtype=torch.float32, device=device)
        self.span = torch.as_tensor(action_q99, dtype=torch.float32, device=device) - self.q01
        self.max_step = torch.as_tensor(profile["max_step"], dtype=torch.float32, device=device)
        self.lower = torch.as_tensor(profile["lower"], dtype=torch.float32, device=device)
        self.upper = torch.as_tensor(profile["upper"], dtype=torch.float32, device=device)
        self.alpha = float(profile["alpha"])

    def __call__(self, requested, execution_state, valid_mask=None, *, preparation=None, execution_scale=None):
        import torch

        if execution_state.shape != (requested.shape[0], 2, requested.shape[-1]):
            raise ValueError("Execution state must contain previous and filtered physical targets")
        scale = self.execution_scale if execution_scale is None else float(execution_scale)
        if self.timeline is not None:
            requested = self.timeline.prepare(requested, preparation, execution_scale=scale)
        else:
            requested = smooth_actions_torch(requested, self.smooth_taps) * scale
        previous, filtered = execution_state[:, 0], execution_state[:, 1]
        physical = (requested + 1.0) * self.span / 2.0 + self.q01
        result = []
        for index in range(requested.shape[1]):
            accepted = accepted_target(
                physical[:, index], previous, filtered, alpha=self.alpha,
                max_step=self.max_step, lower=self.lower, upper=self.upper,
            )
            result.append(2.0 * (accepted - self.q01) / self.span - 1.0)
            previous = filtered = accepted
        chunk = torch.stack(result, dim=1)
        # Padding is a representation convention, never a physical zero target.
        return chunk if valid_mask is None else chunk.masked_fill(~valid_mask.bool().unsqueeze(-1), 0.0)


def execution_state_from_observation(observation):
    """Keep missing legacy controller state explicit instead of estimating it."""
    value = observation.get(EXECUTION_STATE_KEY)
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2, 7) or not np.isfinite(result).all():
        raise ValueError("ARX observation has invalid execution state")
    return result.copy()
