# 0013. Run the sorting cycle as one state machine with a servo-6 grasp check, a scripted re-grasp and a raise after every release

- Status: Accepted (UNTESTED on the robot: only dry runs and unit tests with a fake arm)
- Date: 2026-11-14
- Deciders: team lead (chessnok, diagram 2026-11-14), agent

## Context

The user's diagram, with their notes:

1. Capture the brick and move it to the right bowl.
2. Go to the start (view) position.
3. Detect the cube.
4. Map image coordinates to the table and move to the cube.
5. Start the capture: close the gripper.
6. Check for the "block" error on ID 6, meaning the gripper stalled on a brick. If there is none, retry from step 4, at most 3 times per cube, and otherwise start from the beginning.
7. Move to the photo position.
8. Release the block, take the photo and run detection. When releasing for the photo, release AND raise the arm.
9. Go back to step 1.

Other facts:
- **Photo pose:** taught by hand (`photo` in `configs/poses.yaml`, [-8.55, 51.63, -40.66, 80.44, -27.73, 3.02]). FK puts its jaw tips on the table, around (295, 23, -5) mm, and the gripper is not straight down. The user requires wrist_roll (motor 5) to be reached exactly there.
- **Detector:** PatchCore plus the mark classifier, both on the NPU, OR per ADR 0008. The VLM is dropped (log 09:35).
- **Gripper behaviour:**
  - Closed on nothing, it stops at about 2.5 normalized (agent B). On a big brick it stops at about 30-34.
  - Studio's plugin sets the gripper's overload protection on connect: OVERLOAD_TORQUE 25 %, PROTECTION_CURRENT 250.

## Decision

`hackjunction2026.sort_cycle` / `scripts/sort_cycle.py`:

- **States:** OBSERVE, LOCATE, HOVER, GRASP (with ≤ 3 tries, then start over once, then stop), TO_PHOTO, RELEASE_PHOTO (open, then raise `raise_mm`), INSPECT, REGRASP, TO_BOWL, RELEASE_BOWL (open, raise), DONE.
- **Log:** one JSONL log per run plus every frame under `captures/sort_cycle/<time>/`.
- **Every move goes through one `Mover`:**
  - a range check;
  - a clearance check of the whole straight joint-space path (30 mm travel, 2 mm near the table). A move into or out of a taught low pose may go down to that pose's own FK height minus 3 mm. Gripper-only moves are range-checked only;
  - a cap of 10 units/s on the arm joints (4 near the table);
  - a prompt in `--step` mode;
  - then `so101_motion.move_linear`, which stops and holds on lag or Ctrl+C.
- **Grasp check on servo 6:** after closing, read Present_Position, Present_Load (addr 60) and Status (addr 65).
  - **Holding** means the gripper stopped at ≥ `hold_min` (8), OR |load| ≥ 150 with the gripper not fully closed, OR an OVERLOAD / OVERCURRENT bit.
  - While closing and holding, the gripper's RAM Torque_Limit (addr 48) is lowered to 300. After a hold, the command is eased to 3 units past the stop position, so the overload protection does not trip and drop the brick.
  - The original limit is restored at the end. No EEPROM writes.
- **Taught poses (photo, bowls):** reached EXACTLY in joint space.
  - The pre-pose above them changes only shoulder_lift, elbow_flex and wrist_flex: a position-only least-squares raise of the jaw centre, falling back to the line towards `observe`. Pan and wrist_roll are already right, and there is no IK re-solve.
  - The "raise" after a release uses the same method.
- **First grasp:** an interface with two implementations.
  - `ScriptedGrasp` (default): open, straight down to `grasp_mm` with gripper-down IK at the brick's XY, close, check, lift.
  - `ACTGrasp`: the Studio-trained ACT run outside Studio with `physicalai.inference.InferenceModel` on the OpenVINO export.
    - Observation: `{"state": (1, 6) normalized, "images.top|floor|arm": (1, H, W, 3) RGB uint8}`, as Studio's model worker sends it.
    - Chunks of 100 steps, re-planned every 10 at 30 Hz. Every step is speed-capped and table-checked.
    - Loading it outside Studio works: 1.3 s on CPU, 0.14 s per chunk on zero frames. It has not been run on the robot.
- **Re-grasp at the photo spot:** always scripted, back to the exact taught pose with the jaws open.
- **Locating the brick:** the wrist-camera homography when `configs/table_homography.yaml` exists, otherwise the top camera model (3-4 cm off at the edge of reach, log 07:12). The `--target X,Y` option skips it.
- **Refusals:** the cycle refuses to start while any pose is missing; a dry run can use `--placeholder-bowls`. A real run needs `--go`, Studio's environment closed (port and wrist camera free) and one confirmation.

## Alternatives considered

- **Running the cycle inside Studio:** Studio runs one policy, not a scripted multi-step flow with detection in between.
- **Gripper-down IK for every pose:** the taught photo pose is not gripper-down, and the user wants its wrist_roll unchanged.
- **Grasp check from the gripper position only:** a small brick may stop close to the empty value. The load and status bits make it more robust; both thresholds are still unmeasured.

## Consequences

Before the first real run, measure:
1. the gripper value, load and status bits when closed empty and when closed on each brick size;
2. `grasp_mm` for the brick size in use;
3. that holding with Torque_Limit 300 and a 3-unit squeeze does not slip;
4. the `bowl_good` pose (still missing; `bowl_defect` exists);
5. the wrist homography (10-minute brick calibration, ADR 0007) or a better top-camera fit.

First run `--go --step` with one brick and hands clear.
