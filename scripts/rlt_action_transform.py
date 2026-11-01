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


class AcceptedActionTransform:
    """Differentiable normalized chunk -> normalized SDK accepted targets."""

    def __init__(self, profile, action_q01, action_q99, *, device="cpu", execution_scale=1.0,
                 smooth_taps: int = 0):
        import torch

        if profile["contract"] != EXECUTION_CONTRACT:
            raise ValueError(f"Unsupported execution transform: {profile['contract']}")
        self.profile = profile
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

    def __call__(self, requested, execution_state, valid_mask=None):
        import torch

        if execution_state.shape != (requested.shape[0], 2, requested.shape[-1]):
            raise ValueError("Execution state must contain previous and filtered physical targets")
        requested = smooth_actions_torch(requested, self.smooth_taps)
        previous, filtered = execution_state[:, 0], execution_state[:, 1]
        physical = (requested * self.execution_scale + 1.0) * self.span / 2.0 + self.q01
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
