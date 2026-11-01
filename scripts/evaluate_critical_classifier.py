#!/usr/bin/env python3
"""Offline evaluation of the critical-phase classifier (no real robot needed).

Replays the precomputed z_rl samples in temporal order through the exact
runtime pipeline (CriticalPhaseDetector: per-chunk probability, moving average,
hysteresis thresholds) and compares the auto-detected critical intervals
against the manual annotations:

  - per-sample classification report (precision / recall / F1 / accuracy),
    split by train vs held-out validation episodes;
  - per-episode interval comparison: detected vs annotated start/end frames,
    overlap IoU, and detection lag in frames.

Usage:
    python scripts/evaluate_critical_classifier.py \
        --precomputed_zrl outputs/critical_zrl_straw.npz \
        --classifier outputs/critical_classifier_straw.pt \
        --val_episodes 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed_zrl", type=str, default=None,
                        help="NPZ from precompute_critical_zrl.py (dataset evaluation).")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Live 'c'-boundary log from train_rlt_stage2_pi05.py "
                             "(evaluate on REAL robot data; mutually exclusive-ish with precomputed_zrl).")
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--val_episodes", type=int, default=None,
                        help="First N annotated episodes held out; defaults to checkpoint metadata, then 3.")
    parser.add_argument("--threshold_on", type=float, default=0.6)
    parser.add_argument("--threshold_off", type=float, default=0.4)
    parser.add_argument("--smooth_steps", type=int, default=10)
    parser.add_argument("--min_on_chunks", type=int, default=0,
                        help="Debounce: hold ON for at least N updates before an OFF is accepted (0 = off).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Update the detector every N frames and hold its state between updates. "
                             "N > 1 requires consecutive frames from --all_frames precomputation.")
    parser.add_argument("--output", type=str, default=None, help="Write the evaluation report as JSON.")
    return parser.parse_args()


def intervals_from_flags(frames: np.ndarray, flags: np.ndarray) -> list[tuple[int, int]]:
    """Merge a per-sample flag sequence into (start, end) frame intervals."""
    intervals = []
    start = None
    for frame, flag in zip(frames, flags):
        frame = int(frame)
        if flag and start is None:
            start = frame
        elif not flag and start is not None:
            intervals.append((start, frame - 1))
            start = None
    if start is not None:
        intervals.append((start, int(frames[-1])))
    return intervals


def interval_union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted(intervals)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_ratio(detected: list[tuple[int, int]], annotated: list[tuple[int, int]]) -> float:
    """Frame-level IoU between detected and annotated intervals (union of all intervals)."""
    det = interval_union(detected)
    ann = interval_union(annotated)
    if not det or not ann:
        return 0.0
    det_frames = set()
    for start, end in det:
        det_frames.update(range(start, end + 1))
    ann_frames = set()
    for start, end in ann:
        ann_frames.update(range(start, end + 1))
    intersection = len(det_frames & ann_frames)
    union = len(det_frames | ann_frames)
    return intersection / union if union else 0.0


def load_annotation_intervals(raw) -> dict[int, list[tuple[int, int]]]:
    """Accept both formats: {episode: [[start, end], ...]} (precompute NPZ)
    and {"annotations": [{"episode":..., "start":..., "end":...}, ...]}."""
    intervals: dict[int, list[tuple[int, int]]] = {}
    if isinstance(raw, dict) and "annotations" not in raw:
        for episode, items in raw.items():
            intervals[int(episode)] = [tuple(map(int, item)) for item in items]
    else:
        for item in (raw["annotations"] if isinstance(raw, dict) else raw):
            intervals.setdefault(int(item["episode"]), []).append(
                (int(item["start"]), int(item["end"]))
            )
    return intervals


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    positive = labels == 1
    tp = int((predictions & positive).sum())
    fp = int((predictions & ~positive).sum())
    fn = int((~predictions & positive).sum())
    tn = int((~predictions & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"samples": len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main() -> None:
    args = parse_args()
    if args.frame_stride < 1:
        raise ValueError("--frame_stride must be positive")
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    if not args.precomputed_zrl and not args.record_path:
        raise ValueError("Provide --precomputed_zrl or --record_path")
    if args.precomputed_zrl and args.record_path:
        raise ValueError("Provide only one of --precomputed_zrl or --record_path")
    if args.record_path:
        from train_critical_classifier import load_record_log
        features, _ = load_record_log(args.record_path)
        z_rl = features["z_rl"]
        proprio = features["proprio"]
        labels = features["labels"]
        episodes = features["episode"]
        frames = features["frame"]
        # Manual 'c' boundaries ARE the ground truth on real robot data:
        # rebuild per-episode intervals from the expanded labels.
        annotations = {}
        for episode in np.unique(episodes):
            mask = episodes == episode
            order = np.argsort(frames[mask])
            ep_frames = frames[mask][order]
            ep_labels = labels[mask][order]
            intervals = []
            start = None
            for i, lab in enumerate(ep_labels):
                if lab and start is None:
                    start = i
                elif not lab and start is not None:
                    intervals.append((int(ep_frames[start]), int(ep_frames[i - 1])))
                    start = None
            if start is not None:
                intervals.append((int(ep_frames[start]), int(ep_frames[-1])))
            if intervals:
                annotations[int(episode)] = intervals
        val_episodes: set[int] = set()
    else:
        data = np.load(args.precomputed_zrl, allow_pickle=False)
        z_rl = data["z_rl"].astype(np.float32)
        proprio = data["proprio"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        episodes = data["episode"].astype(np.int64)
        frames = data["frame"].astype(np.int64)
        annotations = load_annotation_intervals(json.loads(str(data["annotations"])))
        annotated_episodes = sorted(annotations)
        saved_val = checkpoint.get("validation_episodes")
        if args.val_episodes is None:
            val_episodes = set(saved_val if saved_val is not None else annotated_episodes[:3])
        else:
            val_episodes = set(annotated_episodes[:args.val_episodes])
            if saved_val is not None and val_episodes != set(saved_val):
                raise ValueError("Requested validation episodes disagree with the classifier checkpoint")

    raw_predictions = np.zeros(len(labels), dtype=bool)
    detected_predictions = np.zeros(len(labels), dtype=bool)
    device = torch_device(args.device)
    rows = []
    for episode in sorted(np.unique(episodes)):
        indices = np.flatnonzero(episodes == episode)
        indices = indices[np.argsort(frames[indices])]
        ep_z, ep_p = z_rl[indices], proprio[indices]
        ep_labels, ep_frames = labels[indices], frames[indices]
        consecutive = bool(np.all(np.diff(ep_frames) == 1))
        if args.frame_stride > 1 and not consecutive:
            raise ValueError("Strided evaluation requires consecutive frames; precompute with --all_frames")

        # Start each episode with a fresh detector, including debounce state.
        detector = CriticalPhaseDetector(
            args.classifier, device, threshold_on=args.threshold_on,
            threshold_off=args.threshold_off, smooth_steps=args.smooth_steps,
            min_on_chunks=args.min_on_chunks,
        )
        with torch.inference_mode():
            z_t = torch.as_tensor(ep_z, device=device)
            p_t = (torch.as_tensor(ep_p, device=device) - detector.proprio_mean) / detector.proprio_std
            raw = detector.classifier(z_t, p_t).argmax(dim=-1).cpu().numpy() == 1
        raw_predictions[indices] = raw

        # Simulate the online loop: feed samples in temporal order through the
        # detector (smoothing + hysteresis), as the training script would.
        detected_state = False
        preds = np.zeros(len(indices), dtype=bool)
        for offset in range(0, len(indices), args.frame_stride):
            candidate = detector.update(ep_z[offset], ep_p[offset])
            if candidate is not None:
                detected_state = candidate
            preds[offset:offset + args.frame_stride] = detected_state
        detected_predictions[indices] = preds

        annotated = interval_union(annotations.get(episode, []))
        detected = intervals_from_flags(ep_frames, preds)
        iou = overlap_ratio(detected, annotated)
        split = "record" if args.record_path else "validation" if episode in val_episodes else "train"
        rows.append({"episode": int(episode), "split": split, "consecutive_frames": consecutive,
                     "annotated": annotated, "detected": detected, "iou": iou,
                     "on_lag_frames": int(detected[0][0] - annotated[0][0]) if annotated and detected else None,
                     "off_lag_frames": int(detected[-1][1] - annotated[-1][1]) if annotated and detected else None,
                     "raw": classification_metrics(ep_labels, raw),
                     "detector": classification_metrics(ep_labels, preds)})

    report = {"schema_version": 1, "classifier": str(Path(args.classifier).resolve()),
              "parameters": vars(args), "validation_episodes": sorted(val_episodes),
              "splits": {}, "episodes": rows}
    print("\nPer-sample metrics (raw threshold 0.5; detector holds state between updates):")
    for split in ("record",) if args.record_path else ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        mask = np.isin(episodes, [row["episode"] for row in split_rows])
        metrics = {"episodes": [row["episode"] for row in split_rows],
                   "raw": classification_metrics(labels[mask], raw_predictions[mask]),
                   "detector": classification_metrics(labels[mask], detected_predictions[mask]),
                   "mean_iou": float(np.mean([row["iou"] for row in split_rows]))}
        for boundary in ("on", "off"):
            lags = [row[f"{boundary}_lag_frames"] for row in split_rows
                    if row[f"{boundary}_lag_frames"] is not None]
            metrics[f"{boundary}_lag_episodes"] = len(lags)
            metrics[f"mean_{boundary}_lag_frames"] = float(np.mean(lags)) if lags else None
            metrics[f"mean_absolute_{boundary}_lag_frames"] = float(np.mean(np.abs(lags))) if lags else None
        report["splits"][split] = metrics
        for mode in ("raw", "detector"):
            item = metrics[mode]
            print(f"  {split:10} {mode:8} n={item['samples']} accuracy={item['accuracy']:.4f} "
                  f"precision={item['precision']:.4f} recall={item['recall']:.4f} F1={item['f1']:.4f}")
        print(f"  {split:10} mean IoU={metrics['mean_iou']:.4f} "
              f"ON lag={metrics['mean_on_lag_frames']} OFF lag={metrics['mean_off_lag_frames']} frames")
    print("\nEpisode intervals (positive lag = late):")
    for row in rows:
        print(f"  {row['episode']:2d} {row['split']:10} annotation={row['annotated']} "
              f"detection={row['detected']} IoU={row['iou']:.3f} "
              f"ON lag={row['on_lag_frames']} OFF lag={row['off_lag_frames']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Saved evaluation to %s", output)


def torch_device(name: str):
    return torch.device(name if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
