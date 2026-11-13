# Hardware

Source of truth for physical facts. Edit in place; note the date of each measurement.
Code reads device addresses from `configs/hardware.yaml`; keep both in sync.

## Laptop

| Item | Value |
| --- | --- |
| Model | HP OmniBook Ultra Laptop 14t-kd000 |
| CPU | Intel Core Ultra X9 388H (Panther Lake, Core Ultra Series 3) |
| iGPU | Intel Arc B390 (`/dev/dri/renderD128`) |
| NPU | Panther Lake NPU (`/dev/accel/accel0`), user-space driver from the `intel-npu-driver` snap 1.38.0 |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic |
| RAM / disk | 61 GiB / 937 GB |
| Clock | System time zone is PST (UTC-8); the event is in Helsinki (UTC+2) |

## USB topology (2026-11-13)

Everything robot-related hangs off one external hub chain:
VIA Labs USB3.1 hub (`2109:0822` / `2109:2822`) → Realtek RTS5411 hub (`0bda:0411` / `0bda:5411`) →
two servo boards, RealSense, wrist camera.

## Arms

Both arms use Feetech STS3215 servos (model number 777) on a 1 Mbps bus. Servo boards are WCH CH343
USB-serial adapters (`1a86:55d3`, product "USB Single Serial", kernel driver `cdc_acm`).
SO-101 joint IDs: 1 shoulder_pan, 2 shoulder_lift, 3 elbow_flex, 4 wrist_flex, 5 wrist_roll, 6 gripper.

### Arm A: probably the **leader** (unconfirmed)

- Port: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91030613-if00` (was `/dev/ttyACM0`)
- 6 servos answer, firmware 3.10 on all, supply 5.4 V, 25–27 °C, Status 0 (2026-11-13)
- Already calibrated (homing offsets and position limits written to EEPROM before we got it)
- Evidence for "leader": a full gripper open was invisible to the wrist camera, so arm A is not
  the arm that carries the wrist camera. Confirm visually (handle and trigger = leader).

| ID | Joint | Min | Max | Homing offset | Rest pos (2026-11-13) |
| --- | --- | --- | --- | --- | --- |
| 1 | shoulder_pan | 748 | 3431 | 1642 | 1406 |
| 2 | shoulder_lift | 870 | 3251 | 1550 | 879 |
| 3 | elbow_flex | 840 | 3054 | 1494 | 3050 |
| 4 | wrist_flex | 825 | 3142 | 1908 | 830 |
| 5 | wrist_roll | 189 | 3992 | 975 | 202 |
| 6 | gripper | 1685 | 2961 | -1144 | 1707 |

Other EEPROM values read (all IDs unless noted): P=32, D=32, I=0, max torque 1000,
protection current 300 (ID 2: 500), overload torque 80, mode 0 (position), lock 1,
max voltage limit 12.0 V (ID 2: 8.0 V), min voltage 4.0 V, LED alarm mask 44
(overheat | overcurrent | overload), unload mask 47.

Motion tests (2026-11-13, torque limit 500, speed 400–600 steps/s): gripper 1707 → 1900 → 1711 OK;
gripper full open 1710 → 2922 (stops about 20 steps short of the 2941 target at 50 % torque,
mechanical end), relaxes to 2875 when torque is released.

### Arm B: probably the **follower** (unconfirmed), currently NOT answering

- Port: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D041025-if00` (was `/dev/ttyACM1`)
- The board enumerates, but no servo answers PING on IDs 0–20 at any baud rate
  (1M, 500k, 250k, 115.2k, 57.6k, 38.4k). A blinking red LED was reported. See T-005.

## Cameras

| Camera | ID | Node (changes on replug) | Placement / role |
| --- | --- | --- | --- |
| Intel RealSense D415 | serial 344223023464, `8086:0ad3` | color was `/dev/video10`, IR `/dev/video8` (2026-11-13) | Above / in front of the table, sees the workspace. Role: overhead, TBD |
| USB2.0_CAM1 | `05a3:9230` | `/dev/video4` | Wrist camera on the follower gripper (sees both jaws) |
| Second external camera | n/a | n/a | Required by the guide, not connected yet |
| HP 5MP built-in (RGB + IR) | `30c9:00fb` | `/dev/video0`, `/dev/video2` | Do not use (films people) |

## Permissions

- Serial: user `intel` added to `dialout` (effective after re-login); temporary ACLs were set with
  `setfacl -m u:intel:rw /dev/ttyACM*`, which reset on replug.
- GPU / NPU nodes: `uaccess` ACLs give the logged-in user rw, no group change needed.
