# Troubleshooting

Newest entry first. Never delete entries; set `Status: Fixed` and describe the fix.

Template:

```markdown
## T-NNN. Short symptom
- Date: YYYY-MM-DD · Status: Open | Workaround | Fixed
- Symptom: exact error text / observed behaviour, command that produced it
- Cause: what was actually wrong (or "unknown")
- Fix: what made it work, with commands
- Refs: ADRs, log entries, upstream issues
```

## T-010. Backend `uv sync --extra xpu` dies downloading torch on venue Wi-Fi

- Date: 2026-11-13 · Status: Workaround
- Symptom: `error: Failed to download torch==2.11.0+xpu ... I/O operation failed during extraction
  ... network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s)`.
- Cause: slow, flaky venue Wi-Fi (about 0.6 MB/s measured from download.pytorch.org, plus a Wi-Fi
  switch mid-download). The XPU build needs `torch` (792.5 MiB) and `triton-xpu` (732 MiB).
- Fix / workaround: `UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10`; install the backend with
  `--extra cpu` first (CPU torch is about 180 MB; everything else is already in `~/.cache/uv`) to get
  Studio running, then fetch the XPU build separately. Do not `pkill -f "uv sync ..."` from a shell
  whose own command line contains that pattern: it kills itself (exit 144).

## T-009. `git` not installed

- Date: 2026-11-13 · Status: Fixed
- Symptom: `/bin/bash: git: command not found` while cloning Physical AI Studio.
- Cause: fresh Ubuntu install without git.
- Fix: `sudo apt install git` (git 2.53.0).

## T-008. PyTorch does not see the Arc iGPU, OpenVINO only lists CPU

- Date: 2026-11-13 · Status: Fixed (13:13 laptop clock)
- Symptom: `torch.xpu.is_available() == False` with `UserWarning: XPU device count is zero!`
  (torch 2.15.0+xpu); `openvino.Core().available_devices == ['CPU']`.
- Cause: GPU compute runtime not installed: `libze1`, `libze-intel-gpu1`, `intel-opencl-icd` absent
  (apt candidates on 26.04: libze1 1.28.2-2, libze-intel-gpu1 and intel-opencl-icd 26.05.37020.3-1).
  The NPU user-space driver exists only inside the `intel-npu-driver` snap
  (`/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so`), which non-snap
  programs do not see by default.
- Fix:
  1. `sudo apt-get install -y libze1 libze-intel-gpu1 intel-opencl-icd`, which gives
     `torch.xpu.is_available() == True` ("Intel(R) Graphics [0xb080]") and OpenVINO `['CPU', 'GPU']`.
  2. NPU: point the Level Zero loader at both drivers:
     `ZE_ENABLE_ALT_DRIVERS=/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1,/snap/intel-npu-driver/current/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1`
     gives OpenVINO `['CPU', 'GPU', 'NPU']`, and torch XPU still works (matmul verified).
     **Gotcha:** the variable *replaces* the default driver list. With only the NPU driver in it,
     torch fails with `RuntimeError: No XPU devices are available.`
  `scripts/studio_backend.sh` sets this automatically.
- Refs: README Setup step 1.

## T-007. PyPI install picks `physicalai-train 0.1.0` instead of 0.3.0

- Date: 2026-11-13 · Status: Fixed (by not using PyPI)
- Symptom: `uv pip install --torch-backend=xpu "physicalai-train[xpu]"` installs 0.1.0; forcing
  `==0.3.0` fails: `lerobot[dataset]==0.6.0 depends on torchcodec>=0.3.0,<0.12.0` but only
  `torchcodec>=0.13.0+cpu` is available.
- Cause: `--torch-backend=xpu` routes torchcodec to the PyTorch XPU index, which lacks old versions.
- Fix: install from the upstream lockfile (ADR 0003).

## T-006. Agent cannot run remote installer scripts

- Date: 2026-11-13 · Status: Workaround
- Symptom: Claude Code's permission classifier blocked `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Fix: a human runs such commands (`! <command>` in Claude Code). uv 0.13.8 and Node 26.13.1 (nvm)
  were installed this way.

## T-005. Arm B board enumerates but no servo answers; red LED blinking

- Date: 2026-11-13 · Status: Open
- Symptom: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D041025-if00` opens, but
  `scripts/so101_probe.py` gets no reply on IDs 0–20 at 1M / 500k / 250k / 115.2k / 57.6k / 38.4k
  baud. A blinking red LED was reported on the hardware.
  Later (~12:45 laptop clock) the probe printed `cannot open: (22, 'Invalid argument')`. That was a
  red herring: the board opens fine at 1M / 500k / 115.2k / 57.6k / 38.4k but the CH343 under
  `cdc_acm` rejects the non-standard 250000 baud with EINVAL, and the probe stopped there. The probe
  now skips bauds it cannot open. The real problem is unchanged: no servo answers at 1M.
  Note: around 12:12 the boards re-enumerated and swapped `ttyACM0` / `ttyACM1`; by-id paths stayed
  correct.
  In Studio, adding arm B as SO101 Follower fails: `POST /api/robots/catalog/SO101_Follower/identify`
  returns 400. Identify connects as follower and wiggles the gripper, so it needs answering servos.
  Re-probe at 13:32: still silent.
- Cause: unknown. Candidates: servo power supply off or wrong voltage (overvoltage alarm makes every
  servo blink red, see lerobot issue #1010), bus cable from board to first servo not seated, board
  jumper not in the USB position (Waveshare boards).
- Fix: TBD. Check the PSU label against the servo voltage rating, reseat the 3-pin bus cable,
  check that the Waveshare jumper is on **B** (USB mode, per Waveshare docs), then re-run the probe.
  Note: servos in a voltage or overload alarm usually still answer PING (with error bits set), so
  *zero* answers points more to the bus link (jumper on A, cable, no servo power) than to an alarm.
  LED meaning: see docs/hardware.md, "Servo red LED".
- Refs: <https://github.com/huggingface/lerobot/issues/1010>

## T-004. Robotics AI Suite installer is not for Ubuntu 26.04

- Date: 2026-11-13 · Status: Workaround (not used)
- Symptom: docs require Ubuntu 24.04 + ROS 2 Jazzy; the installer removes `ros-*`, `*realsense*`,
  `*openvino*`, `libze1` and similar packages first.
- Fix: do not run it on the host (ADR 0002).

## T-003. Multi-line paste split a command

- Date: 2026-11-13 · Status: Fixed
- Symptom: `bash: /dev/ttyACM1: Permission denied` after pasting
  `sudo setfacl -m u:$USER:rw /dev/ttyACM0 /dev/ttyACM1`; the terminal wrapped the line and ran
  `/dev/ttyACM1` as its own command.
- Fix: paste commands as one line, or run them one at a time.
- Happened again with a long `apt-get install` line (`libclang-dev: command not found`,
  `intel-opencl-icd: command not found`): only the first part was installed. Keep sudo commands
  short; split long package lists into several commands.

## T-002. Serial port `Permission denied`

- Date: 2026-11-13 · Status: Fixed
- Symptom: `[Errno 13] could not open port /dev/ttyACM0: Permission denied`.
- Cause: `/dev/ttyACM*` is `root:dialout 0660`; user `intel` was not in `dialout`.
- Fix: `sudo usermod -aG dialout $USER` (permanent, after re-login), plus
  `sudo setfacl -m u:$USER:rw /dev/ttyACM0` and the same for `/dev/ttyACM1` for immediate access
  (the ACL resets on replug).

## T-001. "Robot is connected" but nothing robot-like on USB

- Date: 2026-11-13 · Status: Fixed
- Symptom: `lsusb` showed only the built-in HP camera.
- Cause: the USB hub with the robot was not plugged in yet.
- Fix: plugged in the hub; devices appeared at 11:38–11:39 (laptop clock).
