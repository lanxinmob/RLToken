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

# Run a deployment/training episode set after the dry run succeeds.
./rlt stage2 --auto-critical --episodes 200 --vla-only \
  --output outputs/rlt/stage2_arx
```

Use `--vla-only` for the first hardware check. See [CONFIG.md](CONFIG.md) for Stage 2
options.

## Development

```bash
.venv/bin/python -m pytest
```

The vendored LeRobot fork is under `lerobot/`. The project is licensed under Apache-2.0;
see [LICENSE](LICENSE).
