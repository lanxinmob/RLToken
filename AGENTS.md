Scope: all RLT work belongs in this repository, including ARX5 deployment, prefix extraction, Stage 1, critical classifiers, Stage 2 and RLT analysis.

- Use the root `./rlt` entry and `.venv`; modify this repository's `scripts/` and `lerobot/` directly.
- pi0.5 base preparation, supervised training and pure VLA inference live in `../pi05`; RLT reads their weights/data rather than copying them.
- Current local ARX5 operation is documented in `ARX5_README.md`; current handoff is `HANDOFF.md`.
- Keep the README sufficient to operate the project and keep the architecture minimal.
- Make small commits with one-line English messages and keep history linear.
- Remove temporary tests after validation; preserve existing tracked tests, user data and historical experiment evidence.
