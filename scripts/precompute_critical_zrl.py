#!/usr/bin/env python3
"""Pre-compute z_rl (RL token) for critical-phase classifier training.

Samples frames from a teleop dataset according to manual critical-phase
annotations (positive = inside an annotated interval, negative = outside, from
annotated episodes only), then runs the frozen π0.5 (image-only prefix) + the
frozen Stage 1 RLT encoder to produce one z_rl per sampled frame.

Output NPZ (consumed by ``train_critical_classifier.py --precomputed_zrl``):
    z_rl      (N, rlt_hidden_dim) float16
    proprio   (N, dataset_state_dim) float32
    labels    (N,) int64 (1 = critical phase)
    episode   (N,) int64
    frame     (N,) int64 (frame index within episode)

Usage:
    python scripts/precompute_critical_zrl.py \
        --pi05_path ./pi05_base \
        --tokenizer_path ./paligemma_tokenizer \
        --rlt_checkpoint ../outputs/pi05_rlt_stage1/best_checkpoint.pt \
        --dataset_path /path/to/my_dataset_multitask \
        --annotations annotations.json \
        --output outputs/critical_zrl.npz \
        --batch_size 8 --device cuda
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_critical_classifier import load_annotations, sample_dataset_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--cache_path", type=str, default=None,
                        help="Reuse the matching Stage 1 prefix cache instead of running the VLA.")
    parser.add_argument("--all_frames", action="store_true",
                        help="With --cache_path, encode every frame in annotated episodes for temporal evaluation.")
    parser.add_argument("--rlt_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pos_per_interval", type=int, default=200)
    parser.add_argument("--neg_per_episode", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def precompute_from_cache(args: argparse.Namespace, annotations: dict, device: torch.device) -> None:
    """Encode a verified full-dataset Stage 1 cache, preserving native state dimensions."""
    import pyarrow.parquet as pq

    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder

    cache = Path(args.cache_path).resolve()
    dataset = Path(args.dataset_path).resolve()
    meta = json.loads((cache / "meta.json").read_text())
    required = {"schema_version": 2, "complete": True, "image_only": True,
                "embedding_stage": "post_transformer", "language_positions_dropped": True,
                "dtype": "float16", "has_mask": True}
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"Cache {key} must be {expected!r}, got {meta.get(key)!r}")
    if Path(meta["dataset_repo_id"]).resolve() != dataset:
        raise ValueError("Embedding cache belongs to a different dataset")
    if args.pi05_path and Path(meta["pi05_path"]).resolve() != Path(args.pi05_path).resolve():
        raise ValueError("Embedding cache belongs to a different VLA")
    paths = sorted((dataset / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet under {dataset}")
    rows = pq.read_table(paths, columns=["index", "episode_index", "frame_index", "observation.state"])
    rows = rows.sort_by("index").to_pydict()
    count, seq_len, hidden = (int(meta[key]) for key in ("n_samples", "seq_len", "vlm_hidden_dim"))
    if rows["index"] != list(range(count)):
        raise ValueError("Cache requires the complete dataset in contiguous global-index order")
    episode_ids = np.asarray(rows["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(rows["frame_index"], dtype=np.int64)
    proprio = np.asarray(rows["observation.state"], dtype=np.float32)
    if proprio.ndim != 2 or not np.isfinite(proprio).all():
        raise ValueError("Dataset states must be a finite (frames, state_dim) array")
    rng = np.random.default_rng(args.seed)
    selected, targets = [], []
    for episode, intervals in sorted(annotations.items()):
        indices = np.flatnonzero(episode_ids == episode)
        if not len(indices) or not np.array_equal(frame_ids[indices], np.arange(len(indices))):
            raise ValueError(f"Episode {episode} has missing or unordered frames")
        marked = np.zeros(len(indices), dtype=bool)
        for start, end in intervals:
            if not 0 <= start <= end < len(indices):
                raise ValueError(f"Episode {episode}: interval [{start}, {end}] is out of bounds")
            marked[start:end + 1] = True
        if args.all_frames:
            chosen = np.arange(len(indices))
        else:
            positive = [rng.choice(np.arange(start, end + 1), min(args.pos_per_interval, end - start + 1),
                                   replace=False) for start, end in intervals]
            negative = rng.choice(np.flatnonzero(~marked), min(args.neg_per_episode, int((~marked).sum())),
                                  replace=False)
            chosen = np.unique(np.concatenate([*positive, negative]))
        selected.extend(indices[chosen].tolist())
        targets.extend(marked[chosen].astype(np.int64).tolist())
    if not selected:
        raise ValueError("No annotated frames selected")
    indices = np.asarray(selected, dtype=np.int64)
    checkpoint = torch.load(args.rlt_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    trained_cache = checkpoint.get("effective_args", {}).get("precomputed_path")
    if not trained_cache or Path(trained_cache).resolve() != cache:
        raise ValueError("Stage 1 checkpoint was not trained from this cache")
    if checkpoint.get("rlt_architecture") != "paper_v1" or checkpoint.get("image_only") is not True:
        raise ValueError("Expected a paper_v1 image-only Stage 1 checkpoint")
    config_fields = {field.name for field in fields(PI05RLTConfig) if field.init}
    config = {key: value for key, value in checkpoint["config"].items() if key in config_fields}
    config["mode"] = "online_rl"
    config["device"] = str(device)
    rlt_config = PI05RLTConfig(**config)
    if rlt_config.vlm_hidden_dim != hidden:
        raise ValueError("Cache hidden dimension disagrees with Stage 1 encoder")
    encoder = RLTokenEncoder(rlt_config)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.to(device).eval().requires_grad_(False)
    for name, dtype, shape in (
        ("prefix_out.mmap", np.float16, (count, seq_len, hidden)),
        ("prefix_mask.mmap", np.bool_, (count, seq_len)),
    ):
        if (cache / name).stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError(f"Unexpected file size for {cache / name}")
    prefix = np.memmap(cache / "prefix_out.mmap", dtype=np.float16, mode="r", shape=(count, seq_len, hidden))
    masks = np.memmap(cache / "prefix_mask.mmap", dtype=np.bool_, mode="r", shape=(count, seq_len))
    z_rl = np.empty((len(indices), rlt_config.rlt_hidden_dim), dtype=np.float16)
    logger.info("Encoding %d frames from %d episodes; z_dim=%d, proprio_dim=%d, device=%s",
                len(indices), len(annotations), rlt_config.rlt_hidden_dim, proprio.shape[1], device)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start + args.batch_size]
            embedding = torch.from_numpy(np.array(prefix[batch], copy=True)).to(device=device, dtype=torch.float32)
            mask = torch.from_numpy(np.array(masks[batch], copy=True)).to(device)
            if not mask.any(dim=1).all() or not torch.isfinite(embedding).all():
                raise ValueError(f"Invalid prefix embeddings at batch starting {start}")
            encoded = encoder(embedding, mask=mask)
            if not torch.isfinite(encoded).all():
                raise ValueError(f"Non-finite z_rl at batch starting {start}")
            z_rl[start:start + len(batch)] = encoded.cpu().numpy().astype(np.float16)
            if start == 0 or (start // args.batch_size + 1) % 25 == 0 or start + len(batch) == len(indices):
                logger.info("Encoded %d/%d frames (%.1fs)", start + len(batch), len(indices), time.monotonic() - started)
    provenance = {"cache_path": str(cache), "cache_metadata": meta,
                  "rlt_checkpoint": str(Path(args.rlt_checkpoint).resolve()),
                  "rlt_step": int(checkpoint["step"]),
                  "annotations_sha256": hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
                  "state_dim": int(proprio.shape[1]), "all_frames": args.all_frames}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, z_rl=z_rl, proprio=proprio[indices], labels=np.asarray(targets, dtype=np.int64),
                        episode=episode_ids[indices], frame=frame_ids[indices], annotations=json.dumps(annotations),
                        provenance=json.dumps(provenance))
    logger.info("Saved %d samples to %s (%.1f%% positive)", len(indices), output, 100 * np.mean(targets))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.pos_per_interval < 0 or args.neg_per_episode < 0:
        raise ValueError("Sampling counts must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    annotations = load_annotations(args.annotations)
    if not annotations:
        raise ValueError("Annotations file contains no intervals")
    if args.cache_path:
        precompute_from_cache(args, annotations, device)
        return
    if args.all_frames:
        raise ValueError("--all_frames requires --cache_path")
    if not args.pi05_path or not args.tokenizer_path:
        raise ValueError("Provide --pi05_path and --tokenizer_path, or --cache_path")

    samples = sample_dataset_frames(
        args.dataset_path,
        annotations,
        pos_per_interval=args.pos_per_interval,
        neg_per_episode=args.neg_per_episode,
        seed=args.seed,
    )
    logger.info("Sampled %d frames for z_rl precomputation", len(samples))

    # ── Load frozen π0.5 + preprocessor (same path as Stage 2) ───────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05_rlt.vla_compat import (
        extract_embeddings as extract_pi05_embeddings,
    )
    from lerobot.policies.pi05_rlt.vla_compat import resize_with_pad as resize_with_pad_torch
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    pi05_policy = PI05Policy.from_pretrained(pretrained_name_or_path=str(args.pi05_path))
    pi05_config = pi05_policy.config
    pi05_config.validate_features()
    pi05_policy.to(device)
    pi05_policy.eval()
    for parameter in pi05_policy.parameters():
        parameter.requires_grad = False
    pi05_model = pi05_policy.model
    image_features = [k for k in pi05_config.input_features if k.startswith("observation.images.")]
    logger.info("π0.5 loaded and frozen; image features: %s", image_features)

    dataset_path = Path(args.dataset_path)
    dataset = LeRobotDataset(dataset_path.name, root=dataset_path, video_backend="pyav")

    def _to_dict_features(features):
        from lerobot.configs.types import PolicyFeature
        return {
            k: {"type": v.type.name, "shape": list(v.shape)}
            if isinstance(v, PolicyFeature) else v
            for k, v in features.items()
        }

    pi05_config.input_features = _to_dict_features(pi05_config.input_features)
    pi05_config.output_features = _to_dict_features(pi05_config.output_features)

    _tk = Path(args.tokenizer_path).expanduser()
    if _tk.exists():
        pi05_config.text_tokenizer_name = str(_tk)  # the current factory reads the tokenizer from the config

    preprocessor, _ = make_pi05_pre_post_processors(
        config=pi05_config,
        dataset_stats=dataset.meta.stats,
    )
    if hasattr(preprocessor, "steps"):
        preprocessor.steps = [
            step for step in preprocessor.steps
            if step.__class__.__name__ not in ("DeviceProcessorStep",)
        ]

    # ── Load frozen RLT encoder (Stage 1 checkpoint) ─────────────────────
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder

    rlt_config = PI05RLTConfig(mode="online_rl")
    encoder = RLTokenEncoder(rlt_config).to(device)
    ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    logger.info("RLT encoder loaded and frozen (z dim %d).", rlt_config.rlt_hidden_dim)

    # ── Compute z_rl per sampled frame ───────────────────────────────────
    image_resolution = pi05_config.image_resolution
    max_action_dim = pi05_config.max_action_dim
    chunk_size = pi05_config.chunk_size

    def process_frame(frame: dict) -> dict:
        item = preprocessor(frame)
        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1 else v
            for k, v in item.items()
        }

    episode_ids = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)

    z_rl_out = np.zeros((len(samples), rlt_config.rlt_hidden_dim), dtype=np.float16)
    state_dim = int(dataset.meta.features["observation.state"]["shape"][0])
    proprio_out = np.zeros((len(samples), state_dim), dtype=np.float32)
    labels_out = np.zeros(len(samples), dtype=np.int64)
    episode_out = np.zeros(len(samples), dtype=np.int64)
    frame_out = np.zeros(len(samples), dtype=np.int64)

    t0 = time.time()
    for start in range(0, len(samples), args.batch_size):
        batch_meta = samples[start:start + args.batch_size]
        processed = [process_frame(dataset[global_idx]) for global_idx, _ in batch_meta]
        batch = {
            k: torch.utils.data.default_collate([item[k] for item in processed])
            if isinstance(processed[0][k], torch.Tensor) else [item[k] for item in processed]
            for k in processed[0].keys()
        }
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bsize = batch["action"].shape[0]

        images, img_masks = [], []
        for feat_name in image_features:
            if feat_name in batch:
                img = batch[feat_name].float()
                img = img.permute(0, 2, 3, 1)
                img = resize_with_pad_torch(img, *image_resolution)
                img = img * 2.0 - 1.0
                img = img.permute(0, 3, 1, 2)
                images.append(img.to(device))
                img_masks.append(torch.ones(bsize, dtype=torch.bool, device=device))
            else:
                images.append(torch.full((bsize, 3, *image_resolution), -1.0, device=device))
                img_masks.append(torch.zeros(bsize, dtype=torch.bool, device=device))

        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = torch.zeros(bsize, chunk_size, max_action_dim, dtype=torch.float32, device=device)

        with torch.no_grad():
            prefix_img, _, prefix_mask = extract_pi05_embeddings(pi05_model, images, img_masks, tokens, masks, actions,
            chunk_size=chunk_size, max_action_dim=max_action_dim, image_only=True)
            z_rl = encoder(prefix_img.float(), mask=prefix_mask).float()  # (B, z_dim)

        if not torch.isfinite(z_rl).all().item():
            raise ValueError(f"Batch {start // args.batch_size}: z_rl contains NaN/Inf")

        end = start + bsize
        z_rl_out[start:end] = z_rl.detach().cpu().numpy().astype(np.float16)
        for offset, (global_idx, label) in enumerate(batch_meta):
            proprio_out[start + offset] = dataset[global_idx]["observation.state"].numpy().astype(np.float32)
            labels_out[start + offset] = label
            episode_out[start + offset] = episode_ids[global_idx]
            frame_out[start + offset] = frame_ids[global_idx]
        logger.info("z_rl batch %d/%d done (%.1fs elapsed)",
                    start // args.batch_size + 1,
                    (len(samples) + args.batch_size - 1) // args.batch_size,
                    time.time() - t0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        z_rl=z_rl_out,
        proprio=proprio_out,
        labels=labels_out,
        episode=episode_out,
        frame=frame_out,
        annotations=json.dumps(annotations),
        pos_per_interval=args.pos_per_interval,
        neg_per_episode=args.neg_per_episode,
    )
    logger.info("Saved %d z_rl samples to %s (%.1f%% positive)",
                len(samples), output, 100.0 * labels_out.mean())


if __name__ == "__main__":
    main()
