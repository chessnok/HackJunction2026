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

## T-015. Local training on the Arc iGPU hangs at step 0: GPU page fault ("Engine memory CAT error")

- Date: 2026-11-14 · Status: Open
- Symptom: Studio local training job (ACT, 51.6 M params, dataset 431a71f7: 6 episodes, 2448 frames,
  cameras 848×480, 848×480, 1024×768, batch 8) stays `running` at progress 0 with no metrics. Job log ends
  at `Epoch 0/4 0/253 … 0.00it/s` (02:33:37). The trainer process uses one core at 99 % and its
  8 dataloader workers are idle; the GPU compute GT never idles.
- Cause: at 02:33:36, on the first training step, the `xe` driver logged
  `GT0: Engine memory CAT error [18]: class=ccs`, `Engine reset: engine_class=ccs`,
  `Timedout job … in python [<trainer pid>]`, `Xe device coredump has been created`. The GPU job was
  killed by the driver, and the trainer waits forever. Likely a bug in the Arc B390 (Panther Lake) +
  kernel 7.0 `xe` + compute runtime 26.05 + torch 2.11+xpu stack, not in our data. The devcoredump
  expired before it could be saved (the kernel keeps it about 5 min, root-only).
- Workaround: stop the job; train on CPU (job `device: cpu`) or on Modal (blocked by T-014). Retrying
  on XPU is untested. For a bug report, copy `/sys/class/drm/card1/device/devcoredump/data`
  (sudo) within 5 minutes of the next hang.

## T-014. Studio "Deploy and connect" to Modal fails after 2 s with a generic message

- Date: 2026-11-14 · Status: Open (waiting for the Modal credit / spend limit)
- Symptom: Studio → Settings → Training Targets → Modal → Deploy and connect: "Deployment failed or
  timed out. Check the Modal dashboard…". Backend log: `RuntimeError: The Modal deployment command
  failed` (`services/modal_deployment.py:138`) 2 s after `POST /api/remote-trainers/providers/modal/deploy`.
- Cause: Studio runs `python -m modal deploy --env main src/trainer/modal_app.py` with stdout and stderr
  sent to DEVNULL, so the real error is hidden. Running the same command by hand shows:
  `Workspace … has exceeded its spend limit` (the hackathon credit not redeemed yet, or a $0 spend
  limit). Auth, environment `main`, modal 1.6.3 and the app script itself are fine; nothing was
  created in Modal (no apps or volumes in `main`).
- Fix: redeem the Modal credit / raise the workspace spend limit on modal.com, then retry Deploy and
  connect. Studio improvement for our fork: log the deploy command's output instead of discarding it.

## T-013. Every camera preview fails: "no publisher responded within timeout"

- Date: 2026-11-13 · Status: Workaround
- Symptom: after Studio was force-killed, every camera preview in Studio failed with
  `physicalai.capture.errors.CaptureTimeoutError: no publisher responded within timeout`
  (`api/camera.py:135` → `physicalai/capture/transport/_shared_camera.py:326`). Opening the same
  D415 directly with pyrealsense2 at the same mode worked (about 20 fps).
- Cause: cameras are shared between processes via iceoryx2 (shared memory, `/tmp/iceoryx2`,
  `/dev/shm/iox2_*`). `SharedCamera.connect()` only starts a publisher when `_probe_service()` finds
  none. SIGKILLed publishers (T-012) left their services behind, so the probe saw a phantom publisher
  and Studio waited for frames forever. `Node.try_cleanup_dead_nodes()` reported 0 dead nodes because
  the running backend held the stale services open.
- Fix / workaround: stop Studio completely, then (with no process mapping iceoryx2) delete
  `/tmp/iceoryx2` and `/dev/shm/iox2_*`, then start Studio. `scripts/studio_stop.sh` now does this.

## T-012. Studio backend ignores SIGTERM; orphaned camera workers keep cameras busy

- Date: 2026-11-13 · Status: Workaround
- Symptom: stopping the backend (`kill <pid>`) leaves it running for more than 15 s, so it needs
  SIGKILL (happened at 13:30 and 16:15). After that, `python -m physicalai.capture.transport._publisher_worker`
  processes survive as orphans (parent = init), ignore SIGTERM too and keep `/dev/videoN` open, so the
  camera is busy for the next Studio start.
- Cause: upstream Studio / physicalai worker shutdown (not investigated further).
- Workaround: `scripts/studio_stop.sh` (SIGTERM, wait 15 s, SIGKILL, including orphaned workers from
  Studio's venv). `scripts/studio_backend.sh` refuses to start a second backend and kills leftover
  workers before starting.

## T-011. Studio datasets contain no depth; RealSense quality is low

- Date: 2026-11-13 · Status: Open (Studio patch in progress, see ADR 0004 when written)
- Symptom: recorded datasets only have color video features (`observation.images.arm`, 1920×1080×3);
  setting a RealSense camera's output type to `depth` / `both` changes nothing.
- Cause (verified in `junction2026` @ 17714488):
  1. `application/backend/src/utils/camera_factory.py` `_ALLOWED_KWARGS["realsense"]` passes only
     width / height / fps to the capture driver; `output_type` and `depth_range_*` are dropped.
  2. Nothing in Studio or `physicalai`'s recording path calls `RealSenseCamera.read_depth()` /
     `read_rgbd()`, so depth is never captured or written.
  3. Our station: the second D415 enumerates at **480 Mbps (USB 2)**, while the first is at 5000 Mbps.
     It was added to Studio as a plain `usb_camera` ("floor regular", 1920×1080 at 8 fps), so no
     depth and a low frame rate. "top realsense" is a `realsense` camera at 15 fps, color only.
- Fix: Studio patch to record aligned color + depth (on branch `hj/realsense-rgbd` in the
  third_party clone, exported to `patches/physical-ai-studio/`); move the second D415 to a USB 3 port
  and cable; configure both D415 with driver `realsense`.

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

- Date: 2026-11-13 · Status: Fixed (~15:35). The team replaced or renumbered servos and fixed the
  power (the bus once looked dead only because the 12 V supply was unplugged). After that all six IDs
  answered at 12.2 V. The original root cause was not pinned down.
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
