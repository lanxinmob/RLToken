#!/usr/bin/env python3
"""Retrain the paper-style RLT Actor and Twin Critic from an existing replay journal.

The script never opens a robot or camera.  It restores the execution contract
stored with a Stage 2 checkpoint, trains fresh Gaussian Actor and Twin Critic heads, and
writes a resume-compatible Stage 2 checkpoint for later shadow evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rlt_core import load_normalization_stats, quantile_normalize  # noqa: E402
from rlt_action_transform import (  # noqa: E402
    AcceptedActionTransform,
    ActionTimeline,
    EXECUTION_CONTRACT,
)
from train_rlt_stage2_pi05 import (  # noqa: E402
    ReplayBuffer,
    ReplayManager,
    actor_reference,
    atomic_torch_save,
    batch_preparation,
    build_stage2_checkpoint,
    capture_rng_state,
    clamp_actor_action,
    canonical_digest,
    discounted_chunk_return,
    frozen_parameters,
    polyak_update,
    sample_fixed_std_action,
    sample_training_proposal,
)
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig  # noqa: E402
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import (  # noqa: E402
    RLTChunkActor,
    RLTTwinCritic,
)


LOGGER = logging.getLogger("rlt.offline_retrain")


def _load_first_record(path: Path) -> dict:
    with path.open("rb") as stream:
        try:
            return pickle.load(stream)
        except EOFError as exc:
            raise ValueError(f"Replay journal is empty: {path}") from exc


def _copy_journal(source: Path, target: Path, limit: int) -> int:
    source_size = source.stat().st_size
    if limit < 0 or limit > source_size:
        raise ValueError(
            f"Invalid replay_journal_bytes={limit}; source journal has {source_size} bytes"
        )
    remaining = limit
    with source.open("rb") as src, target.open("wb") as dst:
        while remaining:
            block = src.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Replay journal ended before the checkpoint boundary")
            dst.write(block)
            remaining -= len(block)
    return limit


def _timeline_from_record(record: dict, actor_handoff_steps: int) -> ActionTimeline:
    timing = np.asarray(record.get("execution_timing"), dtype=np.float32)
    if timing.shape != (7,):
        raise ValueError("Replay rows must contain a seven-value execution_timing signature")
    return ActionTimeline(
        model_steps=int(timing[1]),
        action_hz=float(timing[2]),
        grid_hz=float(timing[3]),
        tick_hz=float(timing[4]),
        smooth_taps=int(timing[5]),
        start_blend_steps=int(timing[6]),
        actor_handoff_steps=int(actor_handoff_steps),
    )


def _prepare(batch: dict[str, torch.Tensor], *, next_state: bool = False) -> dict[str, torch.Tensor]:
    return batch_preparation(batch, next_state=next_state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True,
                        help="Existing Stage 2 checkpoint whose replay and runtime contract are reused")
    parser.add_argument("--replay-journal", type=Path, default=None,
                        help="Replay journal override, default is next to --source-checkpoint")
    parser.add_argument("--init-from-source", action="store_true",
                        help="Start from the saved Actor, Critic and target weights with fresh optimizers")
    parser.add_argument("--gripper-speed", type=float,
                        help="Effective ARX gripper speed after applying the SDK limit, in m/s")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None,
                        help="Stage 1 checkpoint override, default comes from source provenance")
    parser.add_argument("--stats-path", type=Path, default=None,
                        help="Quantile stats override, default comes from source provenance")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=0,
                        help="Critic updates, 0 means utd_ratio * replay size")
    parser.add_argument("--utd-ratio", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--critic-per-actor", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--bc-weight", type=float, default=10.0,
                        help="beta in -Q + beta * MSE(actor_action, reference_action)")
    parser.add_argument("--reference-dropout", type=float, default=0.5)
    parser.add_argument("--policy-fixed-std", type=float, default=0.01)
    parser.add_argument("--actor-residual-scale", type=float, default=0.05,
                        help="Maximum normalized mean correction per action dimension (0 trains a full-output Actor)")
    parser.add_argument("--actor-action-limit", type=float, default=1.0)
    parser.add_argument("--target-min", type=float, default=0.0,
                        help="Lower TD target bound for terminal 0/1 rewards")
    parser.add_argument("--target-max", type=float, default=1.0,
                        help="Upper TD target bound for terminal 0/1 rewards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.updates < 0 or args.utd_ratio < 1 or args.batch_size < 1:
        raise ValueError("updates must be non-negative, utd-ratio and batch-size must be positive")
    if args.critic_per_actor < 1:
        raise ValueError("critic-per-actor must be positive")
    if not 0.0 <= args.reference_dropout <= 1.0:
        raise ValueError("reference-dropout must be in [0, 1]")
    if args.policy_fixed_std <= 0 or args.actor_action_limit <= 0:
        raise ValueError("policy-fixed-std and actor-action-limit must be positive")
    if not np.isfinite(args.actor_residual_scale) or args.actor_residual_scale < 0:
        raise ValueError("actor-residual-scale must be finite and non-negative")
    if args.target_min > args.target_max:
        raise ValueError("target-min must not exceed target-max")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source_checkpoint_path = args.source_checkpoint.expanduser().resolve()
    if not source_checkpoint_path.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source_checkpoint_path}")
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    provenance = source_checkpoint.get("runtime_provenance") or {}
    if args.init_from_source and (provenance.get("actor_residual_scale", 0) > 0) != (args.actor_residual_scale > 0):
        raise ValueError("Initializing from source requires the same Actor type (residual or full-output)")
    execution_profile = copy.deepcopy(source_checkpoint.get("execution_transform"))
    if args.gripper_speed is not None:
        if not execution_profile or execution_profile.get("contract") != EXECUTION_CONTRACT:
            raise ValueError("--gripper-speed requires an ARX accepted-target execution profile")
        if not np.isfinite(args.gripper_speed) or args.gripper_speed <= 0:
            raise ValueError("--gripper-speed must be finite and positive")
        execution_profile["max_step"][-1] = args.gripper_speed * execution_profile["step_dt_s"]

    source_replay = (args.replay_journal or source_checkpoint_path.parent / "replay_journal.pkl").expanduser().resolve()
    if not source_replay.is_file():
        raise FileNotFoundError(f"Replay journal does not exist: {source_replay}")
    source_bytes = int(source_checkpoint.get("replay_journal_bytes", source_replay.stat().st_size))

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_journal = output_dir / "replay_journal.pkl"
    _copy_journal(source_replay, target_journal, source_bytes)

    first_record = _load_first_record(target_journal)
    actor_handoff_steps = int(source_checkpoint.get("actor_handoff_steps", 0))
    timeline = _timeline_from_record(first_record, actor_handoff_steps)
    if int(np.asarray(first_record["action_chunk"]).shape[0]) != timeline.steps:
        raise ValueError(
            "Replay action horizon does not match its execution timing: "
            f"rows={np.asarray(first_record['action_chunk']).shape[0]} timeline={timeline.steps}"
        )

    stage1_path = (args.stage1_checkpoint or provenance.get("rlt_checkpoint"))
    if stage1_path is None:
        raise ValueError("Source checkpoint has no Stage 1 provenance; pass --stage1-checkpoint")
    stage1_path = Path(stage1_path).expanduser().resolve()
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
    stage1_config = stage1_checkpoint.get("config") or {}

    stats_path = args.stats_path or provenance.get("stats_source")
    pi05_path = provenance.get("pi05_path")
    if stats_path is None or pi05_path is None:
        raise ValueError("Source provenance lacks pi05_path/stats_source; pass --stats-path and use a complete source checkpoint")
    stats, stats_source = load_normalization_stats(Path(pi05_path).expanduser(), str(stats_path))

    device = torch.device(args.device)
    buffer_capacity = max(100_000, int(source_checkpoint.get("replay_entries", 0)))
    buffer = ReplayBuffer(buffer_capacity, seed=args.seed, sample_strategy="uniform")
    replay = ReplayManager(buffer, str(target_journal), restore=True, timeline=timeline)
    if len(buffer) < args.batch_size:
        raise ValueError(f"Replay has {len(buffer)} rows, fewer than batch size {args.batch_size}")
    execution_entries = buffer.stats()["execution_entries"]
    if execution_entries < args.batch_size:
        raise ValueError(
            f"Replay has only {execution_entries} rows with execution state, fewer than batch size {args.batch_size}"
        )

    state_dim = int(np.asarray(first_record["proprio"]).shape[-1])
    action_dim = int(np.asarray(first_record["ref_chunk"]).shape[-1])
    z_dim = int(stage1_config.get("rlt_hidden_dim", 2048))
    actor_hidden_dim = int(provenance.get("actor_hidden_dim", 256))
    actor_num_layers = int(provenance.get("actor_num_layers", 2))
    critic_hidden_dim = int(provenance.get("critic_hidden_dim", 256))
    critic_num_layers = int(provenance.get("critic_num_layers", 2))

    rlt_config = PI05RLTConfig(
        device=str(device),
        mode="online_rl",
        rlt_hidden_dim=z_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        actor_hidden_dim=actor_hidden_dim,
        actor_num_layers=actor_num_layers,
        critic_hidden_dim=critic_hidden_dim,
        critic_num_layers=critic_num_layers,
        policy_fixed_std=args.policy_fixed_std,
        actor_residual_scale=args.actor_residual_scale,
        discount=args.discount,
        bc_weight=args.bc_weight,
        target_tau=args.target_tau,
        action_stride=2,
        n_action_steps_rl=10,
        max_state_dim=max(32, state_dim),
        max_action_dim=max(32, action_dim),
    )

    action_q01 = torch.as_tensor(stats["action"]["q01"], dtype=torch.float32, device=device)
    action_q99 = torch.as_tensor(stats["action"]["q99"], dtype=torch.float32, device=device)
    if action_q01.numel() != action_dim or action_q99.numel() != action_dim:
        raise ValueError("Action normalization statistics do not match replay action dimension")

    action_transform = None
    if execution_profile is not None:
        if execution_profile.get("contract") != EXECUTION_CONTRACT:
            raise ValueError(f"Unsupported execution contract: {execution_profile.get('contract')}")
        action_transform = AcceptedActionTransform(
            execution_profile,
            action_q01,
            action_q99,
            device=device,
            execution_scale=float(source_checkpoint.get("execution_scale", 1.0)),
            smooth_taps=timeline.smooth_taps,
            timeline=timeline,
        )

    critic_proprio_dim = state_dim + (2 * action_dim if action_transform is not None else 0)
    actor = RLTChunkActor(
        rlt_config,
        hidden_dim=actor_hidden_dim,
        num_layers=actor_num_layers,
    ).to(device)
    actor.fixed_std = args.policy_fixed_std
    actor_target = copy.deepcopy(actor).to(device)
    critic = RLTTwinCritic(
        rlt_config,
        hidden_dim=critic_hidden_dim,
        num_layers=critic_num_layers,
        proprio_dim=critic_proprio_dim,
        action_steps=timeline.steps,
    ).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    if args.init_from_source:
        actor.load_state_dict(source_checkpoint["actor_state_dict"])
        actor_target.load_state_dict(source_checkpoint["actor_target_state_dict"])
        critic.load_state_dict(source_checkpoint["critic_state_dict"])
        critic_target.load_state_dict(source_checkpoint["critic_target_state_dict"])
    for parameter in actor_target.parameters():
        parameter.requires_grad = False
    for parameter in critic_target.parameters():
        parameter.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    def critic_proprio(batch: dict[str, torch.Tensor], *, next_state: bool = False) -> torch.Tensor:
        state_key = "next_proprio" if next_state else "proprio"
        state = batch[state_key]
        if action_transform is None:
            return state
        execution_key = "next_execution_state" if next_state else "execution_state"
        anchor = quantile_normalize(batch[execution_key], action_q01, action_q99)
        return torch.cat([state, anchor.flatten(start_dim=1)], dim=-1)

    def critic_action(
        batch: dict[str, torch.Tensor],
        action: torch.Tensor,
        *,
        next_state: bool = False,
        reference: bool = False,
    ) -> torch.Tensor:
        preparation = _prepare(batch, next_state=next_state)
        if action_transform is not None:
            execution_key = "next_execution_state" if next_state else "execution_state"
            return action_transform(
                action,
                batch[execution_key],
                batch["valid_mask"],
                preparation=preparation,
                execution_scale=1.0,
                actor=not reference,
            )
        prepared = timeline.prepare(action, preparation, execution_scale=1.0, actor=not reference)
        return prepared.masked_fill(~batch["valid_mask"].bool().unsqueeze(-1), 0.0)

    total_updates = args.updates or args.utd_ratio * len(buffer)
    critic_updates = 0
    actor_updates = 0
    metrics_path = output_dir / "offline_metrics.jsonl"
    LOGGER.info(
        "Offline RLT retrain: rows=%d execution_rows=%d critic_updates=%d batch=%d beta=%.4f",
        len(buffer), execution_entries, total_updates, args.batch_size, args.bc_weight,
    )
    LOGGER.info(
        "Paper loss: actor_loss = -Q(actor_action) + %.4f * MSE(actor_action, replay_reference)",
        args.bc_weight,
    )

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for update in range(total_updates):
            batch = replay.sample_batch(args.batch_size, require_execution_state=action_transform is not None)
            if not batch:
                raise RuntimeError("Replay sampling returned an empty batch")
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.no_grad():
                bootstrap = batch["done"].squeeze(-1) < 0.5
                target_q = torch.zeros_like(batch["done"])
                if bootstrap.any():
                    next_batch = {key: value[bootstrap] for key, value in batch.items()}
                    next_mean, next_std = actor_target(
                        next_batch["next_z_rl"],
                        next_batch["next_proprio"],
                        next_batch["next_ref_chunk"],
                    )
                    next_mean = clamp_actor_action(next_mean, args.actor_action_limit)
                    next_action = clamp_actor_action(
                        sample_fixed_std_action(next_mean, next_std),
                        args.actor_action_limit,
                    )
                    target_q[bootstrap] = critic_target.q_min(
                        next_batch["next_z_rl"],
                        critic_proprio(next_batch, next_state=True),
                        critic_action(next_batch, next_action, next_state=True),
                    )
                chunk_return, bootstrap_discount = discounted_chunk_return(
                    batch["rewards"], batch["valid_mask"], args.discount
                )
                td_target = chunk_return + bootstrap_discount * (1.0 - batch["done"]) * target_q
                td_target = td_target.clamp(min=args.target_min, max=args.target_max)

            q1, q2 = critic(
                batch["z_rl"],
                critic_proprio(batch),
                batch["action_chunk"],
            )
            critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
            critic_optimizer.step()
            critic_updates += 1

            actor_q = float("nan")
            reference_q = float("nan")
            bc_penalty_value = float("nan")
            if critic_updates % args.critic_per_actor == 0:
                rollout_mean, _ = sample_training_proposal(
                    actor, batch, 0.0, args.actor_action_limit,
                )
                with torch.no_grad():
                    actor_q_tensor = critic.q_min(
                        batch["z_rl"],
                        critic_proprio(batch),
                        critic_action(batch, rollout_mean),
                    ).mean()
                    reference_q_tensor = critic.q_min(
                        batch["z_rl"],
                        critic_proprio(batch),
                        critic_action(batch, actor_reference(actor, batch), reference=True),
                    ).mean()
                with frozen_parameters(critic):
                    _, sampled_action = sample_training_proposal(
                        actor, batch, args.reference_dropout, args.actor_action_limit,
                    )
                    sampled_actor_q = critic.q_min(
                        batch["z_rl"].detach(),
                        critic_proprio(batch),
                        critic_action(batch, sampled_action),
                    ).mean()

                    bc_target = batch["reference_targets"]
                    prepared_sample = timeline.prepare(
                        sampled_action,
                        _prepare(batch),
                        execution_scale=1.0,
                        actor=True,
                    )
                    valid_mask = batch["valid_mask"].float()
                    bc_error = (prepared_sample - bc_target).pow(2).mean(dim=-1)
                    bc_penalty = (bc_error * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
                    actor_loss = -sampled_actor_q + args.bc_weight * bc_penalty

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
                actor_optimizer.step()
                actor_updates += 1
                polyak_update(actor, actor_target, args.target_tau)
                polyak_update(critic, critic_target, args.target_tau)

                actor_q = float(actor_q_tensor.detach())
                reference_q = float(reference_q_tensor.detach())
                bc_penalty_value = float(bc_penalty.detach())

            if (update + 1) % args.log_every == 0 or update == 0 or update + 1 == total_updates:
                row = {
                    "update": update + 1,
                    "critic_updates": critic_updates,
                    "actor_updates": actor_updates,
                    "q1": float(q1.mean().detach()),
                    "q2": float(q2.mean().detach()),
                    "q1_max": float(q1.max().detach()),
                    "q2_max": float(q2.max().detach()),
                    "td_target": float(td_target.mean().detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "actor_q": actor_q if np.isfinite(actor_q) else None,
                    "reference_q": reference_q if np.isfinite(reference_q) else None,
                    "bc": bc_penalty_value if np.isfinite(bc_penalty_value) else None,
                }
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                metrics_file.flush()
                LOGGER.info(
                    "update=%d/%d critic=%d actor=%d q1=%.4f q2=%.4f target=%.4f actorQ=%s refQ=%s bc=%s",
                    row["update"], total_updates, critic_updates, actor_updates,
                    row["q1"], row["q2"], row["td_target"],
                    "" if not np.isfinite(actor_q) else f"{actor_q:.4f}",
                    "" if not np.isfinite(reference_q) else f"{reference_q:.4f}",
                    "" if not np.isfinite(bc_penalty_value) else f"{bc_penalty_value:.5f}",
                )

    shutil.copyfile(source_checkpoint_path.parent / "hardware_info.json", output_dir / "hardware_info.json") \
        if (source_checkpoint_path.parent / "hardware_info.json").is_file() else None
    runtime_provenance = copy.deepcopy(provenance)
    if args.gripper_speed is not None:
        runtime_provenance["hardware_config"]["gripper_speed"] = args.gripper_speed
    runtime_provenance.update(actor_residual_scale=args.actor_residual_scale, reference_dropout=args.reference_dropout)
    runtime_provenance.pop("digest", None)
    runtime_provenance["digest"] = canonical_digest(runtime_provenance)
    journal_bytes = target_journal.stat().st_size
    checkpoint = build_stage2_checkpoint(
        episode=int(source_checkpoint.get("episode", 0)),
        actor_state_dict=actor.state_dict(),
        actor_target_state_dict=actor_target.state_dict(),
        critic_state_dict=critic.state_dict(),
        critic_target_state_dict=critic_target.state_dict(),
        actor_optimizer=actor_optimizer.state_dict(),
        critic_optimizer=critic_optimizer.state_dict(),
        policy_fixed_std=actor.fixed_std,
        episode_rewards=list(source_checkpoint.get("episode_rewards", [])),
        episode_interventions=list(source_checkpoint.get("episode_interventions", [])),
        total_updates=critic_updates,
        critic_updates=critic_updates,
        actor_updates=actor_updates,
        execution_contract=source_checkpoint.get("execution_contract"),
        execution_transform=execution_profile,
        execution_scale=float(source_checkpoint.get("execution_scale", 1.0)),
        critic_proprio_dim=critic_proprio_dim,
        aligned_critic_updates=critic_updates if action_transform is not None else 0,
        execution_timing=timeline.profile(),
        actor_handoff_steps=actor_handoff_steps,
        critic_action_steps=timeline.steps,
        runtime_provenance=runtime_provenance,
        rng_state=capture_rng_state(),
        replay_rng_state=copy.deepcopy(buffer._rng.bit_generator.state),
        replay_journal_bytes=journal_bytes,
        replay_entries=replay.stats()["adds_total"],
    )
    atomic_torch_save(checkpoint, output_dir / "final_checkpoint.pt")
    (output_dir / "offline_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Saved offline Stage 2 checkpoint to %s with %d replay rows",
        output_dir / "final_checkpoint.pt", replay.stats()["adds_total"],
    )


if __name__ == "__main__":
    main()
