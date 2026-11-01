"""Local ARX5 RLT stages; pi0.5 supervised training/inference live in ../pi05."""

import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PI05 = ROOT.parent / "pi05"
OUTPUT = ROOT / "outputs/rlt"
BASE = PI05 / "outputs/pi05/base"
DATASET = PI05 / "outputs/pi05/dataset"
VLA = PI05 / "checkpoints/28000steps/pretrained_model"
ENCODER = OUTPUT / "stage1_pi05_28000/best_checkpoint.pt"


def run(command):
    subprocess.run([str(value) for value in command], cwd=ROOT, check=True)


def check(gpu):
    import torch
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder, RLTChunkActor, RLTTwinCritic

    assert all(callable(cls) for cls in (RLTokenEncoder, RLTChunkActor, RLTTwinCritic))
    print(json.dumps({"python": sys.version.split()[0], "rlt_root": str(ROOT),
                      "rlt_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      "lerobot": importlib.metadata.version("lerobot"),
                      "pi05_checkpoint": str(VLA), "encoder": str(ENCODER),
                      "hardware_controllers_created": False}, indent=2), flush=True)
    if gpu:
        assert torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
        x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        assert torch.all(x @ x == 32).item()
        print("CUDA BF16 PASS:", torch.cuda.get_device_name())


def precompute(args):
    run([sys.executable, ROOT / "scripts/precompute_pi05_embeddings.py",
         "--pi05_path", args.pi05_path, "--tokenizer_path", BASE / "tokenizer",
         "--dataset_repo_id", DATASET, "--output_dir", args.output,
         "--camera_map", '{"observation.images.cam_wrist":"observation.images.cam_wrist"}',
         "--batch_size", "2", "--num_workers", "0", "--device", "cuda",
         "--shard_episodes", args.episodes])


def stage1(args):
    command = [sys.executable, ROOT / "scripts/train_rlt_stage1_pi05.py",
               "--precomputed_path", args.embeddings.resolve(), "--output_dir", args.output,
               "--steps", args.steps, "--batch_size", args.batch_size, "--lr", "1e-4",
               "--warmup_steps", min(200, args.steps), "--save_every", args.save_every or args.steps,
               "--log_every", "1", "--num_workers", "0", "--device", "cuda", "--gpus", "0", "--use_amp"]
    if args.dry_run:
        command.append("--dry_run")
    run(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("check")
    doctor.add_argument("--gpu", action="store_true")
    embedding = commands.add_parser("precompute")
    embedding.add_argument("--pi05-path", type=Path, default=VLA)
    embedding.add_argument("--output", type=Path, default=OUTPUT / "embeddings_pi05_28000")
    embedding.add_argument("--episodes", default="")
    token = commands.add_parser("stage1")
    token.add_argument("--embeddings", type=Path, default=OUTPUT / "embeddings_pi05_28000")
    token.add_argument("--output", type=Path, default=OUTPUT / "stage1_pi05_28000")
    token.add_argument("--steps", type=int, default=3)
    token.add_argument("--dry-run", action="store_true")
    token.add_argument("--batch-size", type=int, default=1)
    token.add_argument("--save-every", type=int, default=0)
    grippers = commands.add_parser("gripper-check")
    grippers.add_argument("--config", type=Path, default=ROOT / "rlt_stage2.json")
    grippers.add_argument("--seconds", type=float, default=60)
    grippers.add_argument("--output", type=Path, default=OUTPUT / f"gripper_check_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    for name in ("stage2-check", "stage2"):
        online = commands.add_parser(name)
        online.add_argument("--config", type=Path, default=ROOT / "rlt_stage2.json")
        online.add_argument("--checkpoint", "--pi05-path", dest="checkpoint", type=Path, default=VLA)
        online.add_argument("--rlt-checkpoint", type=Path, default=ENCODER)
        online.add_argument("--auto-critical", action="store_true")
        online.add_argument("--critical-classifier", type=Path)
        if name == "stage2":
            online.add_argument("--episodes", type=int, default=200)
            online.add_argument("--steps-per-episode", type=int,
                                help="Maximum control steps per episode (default from config)")
            online.add_argument("--output", type=Path, default=OUTPUT / f"stage2_arx_{datetime.now():%Y%m%d_%H%M%S}")
            online.add_argument("--dry-run", action="store_true")
            online.add_argument("--resume-from", type=Path)
            online.add_argument("--start-blend-steps", dest="start_blend_steps", type=int,
                                help="Ease from the measured pose over N ticks at episode start")
            online.add_argument("--vla-only", dest="vla_only", action="store_true",
                                help="Deployment validation: VLA executes, no actor/critic updates")
            online.add_argument("--exec-grid-hz", dest="exec_grid_hz", type=float, default=None,
                                help="Execution-only dense grid; requires --vla-only "
                                     "(100 => ~3.3x denser points at the same tick rate)")
            online.add_argument("--no-preview", action="store_true",
                                help="Disable the live OpenCV camera window")
            online.add_argument("--async-inference", "--async_inference",
                                dest="async_inference", action="store_true",
                                help="Plan the next VLA chunk in a background thread instead of "
                                     "blocking the control loop at every chunk boundary")
            online.add_argument("--no-rtc-enabled", "--no-rtc_enabled",
                                dest="rtc_enabled", action="store_false",
                                help="Disable RTC guidance; required by --async-inference")
            online.add_argument("--rtc-mode", "--rtc_mode", dest="rtc_mode",
                                choices=["guided", "trained"],
                                help="RTC mode override (ARX Stage 2 uses guided)")
            online.add_argument("--rtc-execution-horizon", "--rtc_execution_horizon",
                                dest="rtc_execution_horizon", type=int,
                                help="RTC execution horizon override")
            online.add_argument("--rtc-max-guidance-weight", "--rtc_max_guidance_weight",
                                dest="rtc_max_guidance_weight", type=float,
                                help="RTC hard-pin weight; higher suppresses plan-to-plan "
                                     "boundary disagreement harder (default 5.0)")
            online.add_argument("--actor-execute-mean", "--actor_execute_mean",
                                dest="actor_execute_mean", action="store_true",
                                help="Execute the actor's mean action instead of a fixed-std sample; "
                                     "removes per-step exploration noise from the accepted targets")
    args = parser.parse_args()
    if args.command == "check":
        check(args.gpu)
    elif args.command == "precompute":
        precompute(args)
    elif args.command == "stage1":
        if args.steps < 1:
            parser.error("steps must be positive")
        stage1(args)
    elif args.command == "gripper-check":
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            parser.error("seconds must be positive and finite")
        from rlt_stage2 import gripper_check
        gripper_check(args)
    else:
        from rlt_stage2 import run as stage2_run
        if args.command == "stage2-check":
            with tempfile.TemporaryDirectory(prefix="rlt-stage2-preflight-") as output:
                stage2_run(SimpleNamespace(**vars(args), output=Path(output), episodes=200, dry_run=True))
        else:
            if args.episodes < 1:
                parser.error("episodes must be positive")
            stage2_run(args)


if __name__ == "__main__":
    main()
