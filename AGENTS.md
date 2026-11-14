# Agent rules: HackJunction2026

Rules for AI coding agents (Claude Code, Codex, Copilot, ...) **and** humans working in this repo.
Read this file completely before changing anything. `CLAUDE.md` imports it.

## Project in one paragraph

Junction 2026 "Intel Physical AI Challenge": a closed-loop SO-101 arm system that inspects tabletop
objects, detects defective ones and sorts them (PERCEIVE → DETECT → REASON → EXECUTE → OPTIMIZE).
Hardware: two SO-101 arms (leader + follower, Feetech STS3215 servos), Intel RealSense D415,
wrist USB camera, HP OmniBook Ultra 14 (Core Ultra X9 388H "Panther Lake", Arc B390 iGPU, NPU,
Ubuntu 26.04). Stack: Intel Physical AI Studio (`junction2026` branch), Anomalib, OpenVINO,
PyTorch XPU, Modal for VLA training.

- Challenge guide and progress checklist: [docs/challenge.md](docs/challenge.md)
- Hardware facts (source of truth): [docs/hardware.md](docs/hardware.md)
- Why things are the way they are: [docs/decisions/](docs/decisions/)

## Language

Everything committed is **English**: code, comments, docs, logs, commit messages.
Team chat can be in any language; translate when you write it down.

## Record keeping (mandatory)

Write things down **when they happen**, in the same commit as the change. If it is not in the
repo, the next person (or agent) does not know it.

| What | Where | When |
| --- | --- | --- |
| Decision | `docs/decisions/NNNN-kebab-title.md` (ADR, copy `0000-template.md`), add to the index in `docs/decisions/README.md` | Choosing or rejecting a library, version, model, approach, hardware layout; anything a teammate could ask "why?" about |
| Error / problem | `docs/troubleshooting.md`, new entry **at the top** | Any error that cost more than 5 minutes, can recur, or needed a workaround |
| What happened | `docs/log/YYYY-MM-DD.md` (one file per day, append only) | Every work session: timestamped bullets of what was done, the result, what is next |
| Hardware fact | `docs/hardware.md` (edit in place) | Ports, IDs, serial numbers, calibration, power, wiring, camera placement |
| Device addresses for code | `configs/hardware.yaml` | Whenever `docs/hardware.md` changes something code depends on |
| How to run things | `README.md` | When a setup step or command changes |
| Challenge progress | `docs/challenge.md` checklist | When a step is done or blocked |

Rules:

- **Decisions are append-only.** To change one, write a new ADR that supersedes it and set the
  old one's status to `Superseded by NNNN`. Never silently edit history.
- **Troubleshooting entries are never deleted.** When fixed, set `Status: Fixed` and write the fix.
  Always include the exact error text, versions, and the command that produced it.
- **Log entries are facts, not plans**: what ran, what it printed, what changed. Link ADR /
  troubleshooting IDs instead of repeating them. Timestamps use the laptop clock and say so.
- Prefer exact values (register values, versions, commit SHAs, error strings) to prose.
- Mark anything not verified on real hardware as **untested**.
- Never record secrets, tokens, personal data, or images of people. Camera frames stay out of git
  (`captures/` is ignored).

## Robot safety (hard rules)

1. Read-only bus access (PING / READ, `scripts/so101_probe.py`) is always allowed.
2. Any motion needs an explicit request from a human in the current session. Before moving, say
   which arm, which joint, how far and how fast, and ask people to keep their hands clear.
3. First motion on a new setup or joint: smallest useful move, low speed, limited torque
   (`scripts/so101_move.py` defaults). Gripper / wrist roll before shoulder / elbow.
4. Torque is switched off in a `finally` block. Never leave a process holding torque unattended.
5. No EEPROM writes (ID, baud rate, position limits, homing offset, PID, voltage limits, lock)
   without explicit human approval. Log every such write and update `docs/hardware.md`.
   `hackjunction2026.feetech.Bus.write` refuses EEPROM addresses on purpose.
6. A servo with a non-zero `Status` register: stop, record it in `docs/troubleshooting.md`, then
   continue.
7. Never assume the supply voltage. 7.4 V servos on a 12 V supply go into overvoltage alarm
   (blinking red LED) and can be damaged.
8. Any closed-loop / policy run must have a working stop path (kill switch, Ctrl+C handler that
   disables torque) before it is started.

## Environment

- Python is managed with **uv**. Agents do not run `sudo` or `curl | sh` installers; they print the
  command and a human runs it (in Claude Code: `! <command>`).
- Physical AI Studio lives in `third_party/physical-ai-studio` (git clone of the `junction2026`
  branch, **not committed**, see README). Its backend and UI have their own lockfiles; use
  `uv sync` / `npm install` there, never `pip install` into its env (ADR 0003).
- Anomalib never goes into Studio's env (its PyTorch must not clash with Studio's). It is the
  `anomalib` uv extra of this repo: `uv sync --extra capture --extra anomalib` (ADR 0004).
- Project tools: `uv run scripts/<tool>.py` from the repo root.
- Serial ports: use `/dev/serial/by-id/...` from `configs/hardware.yaml`, never `/dev/ttyACM*`
  (numbering changes on replug).
- `/dev/videoN` numbers change on replug: resolve cameras by name or serial.
- Never use the laptop's built-in HP camera in scripts: it films people.

## Git

- Small commits, imperative subject of at most 72 chars, e.g. `docs: record follower power issue`.
- Never commit datasets, recordings, checkpoints, exported models, `.venv`, `third_party/`,
  camera frames, `.env`, or tokens (HF, Modal).
- No force-push to `main`, no rewriting shared history.

## Definition of done

It works on the real hardware (or the doc says why it was not tested), the records above are
updated, and it is committed.
