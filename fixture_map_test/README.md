# fixture_map_test

Turns YOLO fixture detections into map-frame landmarks with uncertainty, saved
image patches, and RViz markers.

**This is stage-1, pre-robot, tripod testing.** There is no robot here and no
pose estimation. A static transform stands in for localization: you measure
where the tripod is, tell the launch file, and the node treats that as truth.
The question this package answers is narrow and worth answering on its own —
*given a detection and a known fixture size, does the fixture land in the right
place on the map?* Everything harder waits until that is yes.

## Layout

```
fixture_map_test/
  fixture_geometry.py    ROS-free. Pinhole range, uncertainty, back-projection.
  fixture_registry.py    ROS-free. Association policy against the apriori map.
  markers.py             MarkerArray construction.
  fixture_map_node.py    Thin ROS wrapper: messages, TF, patches, logging.
config/fixture_classes.yaml   Real fixture sizes, per YOLO class.
config/apriori_map.yaml       Where CAD says the fixtures are.
launch/tripod_test.launch.py  Two static TFs plus the node.
scripts/check_frames.py       Standalone frame + range checker. No ROS.
test/test_core.py             pytest over the two ROS-free modules.
```

The split is deliberate: the geometry and the association policy import nothing
from ROS, so the interesting decisions can be tested exhaustively on a laptop
with no hardware, no graph and no camera. The node contributes plumbing only.

## The four things you must supply

Nothing in this package works until these four are real numbers rather than the
placeholders shipped in the config files.

| # | What | Where | How to get it | What goes wrong if it's off |
|---|------|-------|---------------|------------------------------|
| 1 | **Fixture sizes** (`width_m`) | `config/fixture_classes.yaml` | The extent the **detector's box** encloses, not the catalog part dimension. Best obtained empirically — see `check_frames.py --bbox` below. | Range is scaled by exactly the same factor, for every observation of that class, forever. A bias, not noise; more frames will not help. |
| 2 | **Class ID strings** | `config/fixture_classes.yaml` keys | Copy them verbatim out of your YOLO model's class list. Some detectors emit numerals as strings. | Detections are silently skipped with a throttled warning. Presents as "my fixture never appears", not as an error. |
| 3 | **Measured fixture positions** | `config/apriori_map.yaml` | Tape measure or CAD, in the map frame. | Everything gets FLAGGED, or worse, wrong positions get CONFIRMED because the gate happened to be generous. |
| 4 | **Measured tripod pose** | `cam_x … cam_roll` launch args | Tape and a level, to the camera **body** origin (`zed_camera_link`), not the front glass. | A rigid offset on every landmark. Common to all observations, so nothing in the pipeline can detect it. |

## Run order

```bash
# 0. Before anything touches hardware: prove the geometry.
python3 -m pytest test/ -q
python3 scripts/check_frames.py

# 1. Build.
cd <your_ws>
colcon build --packages-select fixture_map_test
source install/setup.bash

# 2. Start the camera and your YOLO node, however you normally do.

# 3. Bring up the mapper with your measured tripod pose.
ros2 launch fixture_map_test tripod_test.launch.py \
    cam_x:=0.0 cam_y:=0.0 cam_z:=1.20 cam_yaw:=0.0 \
    detections_topic:=/yolo/detections \
    image_topic:=/zed/zed_node/left/image_rect_color \
    camera_info_topic:=/zed/zed_node/left/camera_info \
    output_map:=fixture_map_out.yaml
```

If the ZED wrapper is already publishing its own TF tree, add
`publish_optical_tf:=false`. Two publishers on the same child frame make the
tree flip between them at whatever rate they happen to run — it looks like
intermittent noise and is horrible to debug.

## RViz setup

* **Fixed Frame:** `map`
* **MarkerArray** on `/fixture_map_node/landmarks`
* **MarkerArray** on `/fixture_map_node/rays`
* Add **TF** as well; seeing `map → zed_camera_link → …_optical_frame` is half
  the diagnosis when something is wrong.

What the markers mean:

| Marker | Meaning |
|--------|---------|
| Orange **sphere** | Observed-only landmark. Diameter is `2σ` — the sphere *is* the error bar. A sphere because depth-from-known-size gives **position only**; a mesh would assert an orientation nothing measured. |
| Green **cube** | Apriori (CAD) landmark, confirmed by observation. |
| Red **cube** | Apriori landmark **flagged**: an observation disagreed by more than `confirm_radius`. The CAD position was *not* moved. |
| Grey **lines** | Rays from the camera origin to each observation this frame. |
| White text | `id`, observation count, current σ. |

## How to read the output

The rays exist to split one symptom into two causes. Look at the ray before you
look at anything else.

* **The ray points at the real fixture, but the marker sits too far along it
  (or not far enough)** → the *bearing* is right and the *range* is wrong. This
  is the depth estimator, which in practice means `width_m`. Bearing needs only
  `(u - cx) / fx`, which calibration gives you; range needs a physical size a
  human typed in. Run `check_frames.py --bbox … --true-range …` and use the
  `width_m` it prints.

* **The ray points somewhere else entirely** → the *bearing* is wrong. No amount
  of `width_m` tuning will fix it. Look at frames: the tripod pose, the
  body→optical transform, whether something else is publishing the optical
  frame too.

* **The marker is in the right place** → stage 1 passes. Move on.

The per-detection log prints the optical point and the map point side by side
for exactly this reason:

```
valve_wheel p=0.91 bbox=(96x94px) | optical [+0.161 -0.080 +1.850] | map [+1.850 -0.161 +1.280] | z=1.850+/-0.043m via width
```

A wrong number in the **optical** column is a size or detector problem. A right
optical column with a wrong map column is a frame problem. Printing only the map
point hides which of the two you have.

### Calibrating `width_m` against a tape measure

```bash
python3 scripts/check_frames.py \
    --bbox 812 275 96 94 --real-width 0.09 --true-range 1.85 \
    --fx 700 --cx 640 --cy 360
```

Sections 1–3 always run and check the frame conventions with no inputs at all.
Section 4 prints the estimate with its σ, the error against the tape in metres
and percent, and — the number that matters — **the `width_m` that would have
made the estimate exact**. If that number is a plausible extent for whatever
your labellers actually boxed, your frames are fine and your config is wrong.
If it is absurd, stop tuning sizes and go look at the transforms.

## Why range error grows so fast

`Z = f·W/w_px`, so `σ_Z = (Z²/(f·W))·σ_w`. The **square** is the point: range
uncertainty grows with the square of range. Doubling the distance quadruples the
error bar with the detector behaving exactly as well as it was. At 5 m with a
9 cm fixture and 3 px of box noise you are looking at roughly ±12 cm from pixel
noise alone, before any error in `W`. That is why `max_range` exists and
defaults to 6 m, and why observations beyond it are dropped rather than
down-weighted.

## The apriori rule

**CAD positions are never overwritten by observation.** Disagreement produces a
flag, not a silent edit.

An apriori position came from a drawing someone is accountable for. An
observation came from a bounding box scaled by a hand-typed width, at an error
growing with the square of range. When they disagree, the observation is far
more likely to be the wrong one — and quietly averaging it into the CAD entry
would destroy the only independently trustworthy number in the system, while
leaving a map that looks converged. So the residual is recorded, the marker goes
red, and a human decides.

Observed-only landmarks have no such provenance, so those *are* refined, by
inverse-variance weighting (`w = 1/σ²`).

`correspondences()` returns CAD-origin landmarks only by default — feeding an
observed-only position to PnP as a "known" 3D point launders the estimator's own
error into a pose solution that then looks confident.

## Not implemented on purpose

* **Data association is nearest-neighbour, within class.** That is only valid
  while **each class has at most one instance in view**. With two knobs a metre
  apart, a half-metre range error — entirely ordinary at 4 m — attaches the
  observation to the wrong knob and nothing in the registry can notice. **Put
  one fixture per class in the tripod scene.** The real fixes are
  project-and-gate (a pose prior predicts each landmark's *pixel* location, so
  association happens where the bearing is accurate instead of where the range
  is not) or geometric-consistency RANSAC over the whole detection set. Neither
  is meaningful before there is a pose estimate to prior with.

* **No orientation.** Depth-from-known-size gives position only. That is why
  observed landmarks are spheres and why `MESH_RESOURCE` markers are not used —
  they would put an orientation on screen that no measurement produced.
  `MESH_RESOURCE` becomes legitimate once the apriori map carries full 6-DoF
  poses.

* **The obliquity proxy corrects nothing.** It is a warning flag: observed
  aspect over nominal aspect. It is also meaningless for rotationally symmetric
  fixtures — a knob has no detectable long axis, so its projected aspect says
  nothing about how oblique it is. Round classes ship without a
  `nominal_aspect` on purpose.

* **No pose estimation, no robot, no motion.** Stage 1 only.
