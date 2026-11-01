# RL-Token-Pi05

Online RL and deployment tools for a frozen π0.5 VLA on ARX5 or SO-101. The π0.5
checkpoint and dataset live in the sibling `../pi05` project; this repository contains the
RLT models, training scripts, runtime and hardware adapter.

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
./rlt stage2-check --checkpoint ../pi05/checkpoints/28000steps/pretrained_model \
  --rlt-checkpoint outputs/rlt/stage1_pi05_28000/best_checkpoint.pt

# Run a deployment/training episode set after the dry run succeeds.
./rlt stage2 --auto-critical --episodes 200 --vla-only \
  --output outputs/rlt/stage2_arx
```

Use `--vla-only` for the first hardware check. Keep the emergency stop available, verify
CAN ports and camera paths, and run `stage2-check` before enabling actor execution. See
[ARX5_README.md](ARX5_README.md) for hardware setup and [CONFIG.md](CONFIG.md) for all
Stage 2 options.

## Development

```bash
.venv/bin/python -m pytest
```

The vendored LeRobot fork is under `lerobot/`. The project is licensed under Apache-2.0;
see [LICENSE](LICENSE).
