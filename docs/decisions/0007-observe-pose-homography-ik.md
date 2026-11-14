# 0007. Locate bricks with a fixed-pose homography and hover above them with numeric IK

- Status: Proposed (software tested offline only; the hardware procedure in the README is not run yet)
- Date: 2026-11-14
- Deciders: Claude Code agent (hover-above-brick task) for the team; confirm after the hardware test

## Context

Step to automate: from the saved pose `observe` (`configs/poses.yaml`) the wrist camera (`arm`,
USB2.0_CAM1, 1024x768) sees the whole white sheet. Find the brick on the sheet, compute where it is
on the table and move the follower so that the gripper hangs straight down right above it. A grasp
policy takes over from there.

What we have:

- Forward kinematics of the follower (agent B, `src/hackjunction2026/kinematics/`,
  `docs/perception/B-arm-kinematics.md`), about ±5 mm, base frame in mm, table plane z = -2.4 mm.
- Straight joint-space moves (`scripts/so101_goto.py`): every joint linear in normalized units, all
  arriving together; target range check with a margin of 3 units; stops when the arm lags.
- Studio state snapshots (Space key) store the follower joints with a comment, so an operator can
  teach points by teleoperation.
- Not known: the wrist camera's intrinsics and lens distortion, and where exactly the camera sits
  on the gripper (it is not in the URDF).

## Decision

1. **Pixel → table: one homography, valid only in the `observe` pose.** The operator teleoperates
   the gripper over 5-6 bricks of distinct colours, one Studio snapshot each with the comment
   `calib <colour>`; FK of each snapshot gives the brick's table XY. The same bricks are then detected
   in one wrist camera image taken in `observe` and matched by colour. A homography is fitted
   (normalized DLT + Gauss-Newton on the error in mm; RANSAC from 6 points) and its leave-one-out
   residual in mm is reported; `calib-fit` refuses to save above 15 mm RMS. Stored in
   `configs/table_homography.yaml` with the pose, the image size, the date and the follower pose
   measured when the image was taken. `plan` refuses when the arm is more than 1.5 normalized units
   away from that measured pose, or when the saved pose changed since the fit.
2. **Tool point: the centre of the jaw gap at fingertip height (`jaw_centre`).** It lies on the line
   between the jaws, `FIXED_JAW_TIP.x + 1.9 + jaw_gap(q_gripper) / 2` in `gripper_link`, at the fixed
   fingertip's height. Calibration and planning use the same point: in calibration it is the centre of
   the brick the operator holds the open jaws around, in planning it is put over the detected brick.
   The hover height is the height of that point (= the fixed fingertip, the lowest part of the open
   gripper when it points down) above the table; default 60 mm. The gripper keeps its `observe` value
   (normalized 93, jaw gap 53 mm, the most open).
3. **IK: numeric, numpy only** (`src/hackjunction2026/kinematics/ik.py`). Levenberg-Marquardt over the
   5 arm joints, forward-difference Jacobians, 18 seeds (3 lift x 3 elbow x 2 roll branches),
   vectorized over targets x seeds. Conditions: tool point position (3), gripper axis = straight down
   (3 components), jaws closing along a given horizontal direction (1, either sign). They are
   consistent because the lift, elbow and wrist_flex axes are horizontal and parallel and the
   wrist_roll axis is perpendicular to them (checked on the model to 2e-6). Kept: converged
   (< 0.5 mm, < 0.5°), elbow up (elbow above the shoulder→wrist line), all arm joints inside
   [-97, 97] normalized; among those, the smallest largest joint change from the start pose (that
   is what sets so101_goto's move time). Jaws close across the brick's short side (its long axis
   + 90°, mapped from the image through the homography); for square bricks (sides within 25 %) both
   directions are tried.
4. **Path check before printing a goto command.** `so101_goto` interpolates linearly in normalized
   units, which map linearly to joint angles, so FK at 50 evenly spaced points of the move reproduces
   the swept arm. Checked points: 6 per link along shoulder→elbow→wrist→gripper base→fixed fingertip
   and along the moving jaw, plus the TCP. Every point must stay 30 mm above the table, at every
   sample, or `plan` refuses. The start pose is the measured `observe` pose.

## Alternatives considered

- **Full camera model (intrinsics + hand-eye extrinsics), ray-plane intersection.** Works from any arm
  pose, but needs an intrinsic calibration (checkerboard, distortion) and a hand-eye calibration of
  the camera on the gripper: more equipment and steps, and more parameters to get wrong under
  hackathon time. In one fixed pose the table-to-image map of a pinhole camera *is* a homography, so
  the 8 homography parameters capture everything a full model would for this use. Lens distortion and
  the brick height (the detected blob includes the brick's top and side faces) are absorbed
  approximately because the fit uses the same kind of brick blobs it is later applied to; the
  leave-one-out residual measures what is left. Revisit if the residual is above ~10 mm, or if bricks
  must be found from more than one pose.
- **Homography fitted to the sheet corners with a measured sheet position.** Needs the sheet placed
  at a known spot in the robot frame and two of its corners are hidden by the gripper. Fitting to
  FK points instead ties the map to the same kinematic model the IK uses, so a constant FK error
  partly cancels between calibration and execution.
- **Target LeRobot's TCP (`gripper_frame_link`, agent B's `tcp`).** It sits on the fixed jaw, 8 mm
  from the roll axis: placed over the brick, the fixed jaw comes down on top of the brick.
- **Target the fingertip midpoint (agent B's `gripper_tip`).** Fine with nearly closed jaws, but the
  moving jaw swings out and up as it opens: at the `observe` opening the midpoint is 38 mm off the
  roll axis and 15 mm above the fixed fingertip, i.e. well outside the gap between the jaws.
  `--point gripper_tip` / `tcp` remain available.
- **Closed-form IK.** Possible (planar 2-link plus offsets), but the URDF has several small offsets
  (18 mm lateral chain offset, 2.8° roll offset, tool point off the roll axis that turns with the
  roll); the numeric solver needs no hand derivation and is verified by an FK round trip.
- **Cartesian straight-line motion or a via point above the target.** Would need new motion code
  (not wanted: so101_goto is the one tested motion path). Checking the straight joint move is
  enough so far: every observe → hover move tried stays clear (lowest point is the fingertip at the
  target). If a target is refused by the path check, a via point can be added later.

## Consequences

- The table map is valid only while the arm stands in `observe` as calibrated and the camera mount
  does not move. Re-run `calib-fit` after changing `observe`, re-mounting the wrist camera, moving the
  robot base relative to the table, or recalibrating the arm.
- Accuracy is bounded by FK (±5 mm), how well the operator centred the jaws over each calibration
  brick, the brick height and lens distortion; the leave-one-out residual is the number to watch.
- Reach with the gripper straight down, jaw centre 60 mm above the table: a ring 76-261 mm from the
  shoulder_pan axis (x = 38.8 mm), x = 120-290 mm on the centre line, 1224 cm² for every jaw direction
  (`pick_above.py reach`). Bricks on the far part of the sheet may be out of reach; `plan` refuses
  them (IK: unreachable).
- The path check covers the table only, with 30 mm for the link thickness. It does not cover objects
  on the table (bricks, the plate), the robot's own base, self-collision, or tracking overshoot.
- Detection finds coloured (saturated) bricks only; white, grey or black bricks are not found. Colour
  names come from fixed hue ranges measured on this camera (2026-11-14 snapshots).
- Jaw gap in mm comes from agent B's mesh table and is not validated in mm; it shifts `jaw_centre` by
  half of any gap error.
- Follow-ups: run the README procedure on the robot, record the leave-one-out residual and the hover
  error, then set this ADR to Accepted or supersede it.
