# RL-Token-Pi05

Online RL and deployment tools for a frozen π0.5 VLA.

## Install

```bash
bash setup_arx.sh
cp config.example.json config.json
# Edit config.json with the local ARX SDK path and Python executable.
```

The ignored `config.json` is local machine configuration. Set the camera and model paths in
`rlt_stage2.json` or pass another JSON file. Keep checkpoints and datasets outside Git.

## Check and run

```bash
./rlt check
PI05_CHECKPOINT=/path/to/pi05/pretrained_model
RLT_CHECKPOINT=/path/to/rlt/best_checkpoint.pt
./rlt stage2-check --checkpoint "$PI05_CHECKPOINT" \
  --rlt-checkpoint "$RLT_CHECKPOINT"

# Check frozen VLA execution after the dry run succeeds.
./rlt stage2 --auto-critical --episodes 200 --vla-only \
  --output outputs/rlt/stage2_arx

# Online Actor/Critic training.
./rlt stage2 --auto-critical --episodes 200

# Continue to episode 200, including previously completed episodes.
./rlt stage2 --auto-critical --episodes 200 \
  --resume-from outputs/rlt/YOUR_RUN/final_checkpoint.pt
```

Online training and `--vla-only` share the configured execution pipeline: 30 Hz model
actions, a 100 Hz interpolation grid, 30 Hz command submission, full-plan 5-tap smoothing,
RTC prefetch and a 3-command startup blend. Ten model actions become 31 commands; the
Critic, replay masks and reward discounting use those actual command ticks.

The first resume from an older execution timeline keeps the Actor, its optimizer and episode
history, resets the Critic and its optimizer, and retains old replay for BC only. Actor takeover
waits for fresh aligned data and updates. Subsequent resumes on the same timeline restore both
networks normally. Use a new output directory for each run; original checkpoints are preserved.

See [ARX5_README.md](ARX5_README.md) for local controls and training behavior, and
[CONFIG.md](CONFIG.md) for deployment options.

## Development

```bash
.venv/bin/python -m pytest
```

The vendored LeRobot fork is under `lerobot/`. The project is licensed under Apache-2.0;
see [LICENSE](LICENSE).
