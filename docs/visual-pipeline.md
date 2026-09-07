# Visual pick-and-place pipeline

The visual reference pipeline estimates every source pose from a fixed overhead
RGB-D camera. A second, oblique camera observes the carried part above the
assembly plate and closes small XY errors before the final vertical descent. It
follows the perception sequence used by the hardware workspace:

```text
RGB image -> GroundingDINO -> SAM -> FoundationPose -> world pose
                                                    -> pick-and-place executor
```

Before each pick, the Panda returns to a fixed joint configuration with its tool
above the robot-side edge of the loose-part workspace at approximately
`[0.45, -0.35, 0.50]`. Fixing the joint configuration, rather than solving the
Cartesian point again, prevents the redundant Panda arm from choosing an elbow
pose that covers the remaining parts. The vision service then processes the
newest RGB-D frame, returns the matching source-part pose, and the executor runs
the normal pick-and-place state machine. Transfers use MoveIt; grasp approach,
lift, placement and retreat are Cartesian paths.

Camera rendering is on demand when the supplied visual executor is active. The
source service requests an overhead frame after the arm stops at the observation
pose. During placement, the executor requests an oblique frame while holding the
part above its target and applies up to two Cartesian XY corrections. Every
request rejects frames older than the request itself. MuJoCo does not render
camera images during a trajectory, which keeps image generation from stalling
the physics and controller loop. With `executor:=none`, RGB-D remains a regular
stream for external policies.

RGB, metric depth and camera calibration from one capture share one timestamp.
The vision service refuses to infer until the RGB and depth timestamps match,
which prevents a visual-servo correction from combining colour from one arm
pose with depth from another.

## Observation boundary

The production pipeline runs with `observation:=rgbd`. In this mode the bridge
does not publish ground-truth part poses or MuJoCo segmentation. The vision node
subscribes only to the RGB image, metric depth image, camera calibration and
public product goal. Target assembly poses come from the input product YAML.

Fixed cameras are used instead of a wrist camera because the arm can leave the
source area unobstructed while perception runs, and a stable extrinsic avoids a
moving-camera calibration dependency. Their optical frames are `realsense` and
`placement_camera`; both transforms are published by the benchmark launch file.

## Production backend

The `foundationpose` backend uses GroundingDINO Tiny, SAM ViT-H and
FoundationPose. FoundationPose is not bundled because it has its own build,
CUDA and licensing requirements.

1. Install an NVIDIA driver and a CUDA-compatible build of PyTorch.
2. Install FoundationPose from its upstream repository, including its CUDA
   extensions and model weights.
3. Install the remaining Python packages in the benchmark environment:

   ```bash
   source enter_sim_env.sh
   python -m pip install -r requirements-vision.txt
   export FOUNDATIONPOSE_DIR=/absolute/path/to/FoundationPose
   ```

The first run downloads the GroundingDINO and SAM weights unless they are
already present in the Hugging Face cache. Keep `FOUNDATIONPOSE_DIR` exported in
every terminal that starts the visual pipeline.

Check the local installation without downloading weights or starting ROS:

```bash
./run_visual_pipeline.sh --check-vision
```

The command verifies the FoundationPose source tree, imports its estimator,
checks the Python packages, PyTorch build and CUDA availability. A successful
preflight confirms that the adapter
can be loaded; the complete validation is still a live RGB-D episode because it
also exercises model weights, camera data and the compiled FoundationPose
extensions.

Run a complete visual episode with:

```bash
./run_visual_pipeline.sh \
  --product examples/products/traffic_light.yaml \
  --seed 42 \
  --output-dir runs/visual-traffic-light-42 \
  --headless
```

FoundationPose registration uses five refinement iterations by default to keep
the perception pause short. Change the `register_iterations` ROS parameter if a
camera or object model needs a longer refinement pass.

The launcher defaults to `--vision-backend auto`. It uses this production
backend when the preflight succeeds and otherwise prints a message before
selecting the geometry backend. For recorded experiments, pass the backend
explicitly so the chosen method is visible in the command and result metadata.

## Simulator smoke-test backend

Machines without CUDA can exercise the RGB-D transport, observation pose,
executor and contact physics with the geometry backend:

```bash
./run_visual_pipeline.sh --vision-backend geometry --headless
```

This backend segments the rendered RGB image by colour and derives position and
yaw from metric depth and image geometry. It is an integration test, not a
replacement for GroundingDINO, SAM or FoundationPose, and its results must be
reported as `rgbd_geometry`. The lightweight colour backend skips placement
correction for white parts because the white Panda hand can merge with the
part mask in the oblique view. Source detection and physical placement still
run normally. FoundationPose users may clear
`visual_place_correction_excluded_colors` after validating their masks.

## Outputs

Each run stores the latest annotated frame and machine-readable detections in
`<output-dir>/vision/latest.png` and `<output-dir>/vision/latest.json`. Timestamped
copies make perception failures reproducible. The regular benchmark result and
final physical state remain in the episode directory.

Useful interfaces are:

| Interface | Type | Purpose |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB input |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Metric `32FC1` depth input |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Pinhole intrinsics |
| `/mj_bridge/capture_rgbd` | `std_srvs/Trigger` | Queue one fresh frame after the robot stops |
| `/mj_bridge/capture_placement_rgbd` | `std_srvs/Trigger` | Queue one fresh oblique placement frame |
| `/lego_vision/detect` | `std_srvs/Trigger` | Run perception on the newest frame |
| `/lego_vision/detect_placement` | `std_srvs/Trigger` | Detect the carried part above the assembly area |
| `/lego_bench/vision_detections` | `std_msgs/String` | Detection JSON for logging and external policies |

An external policy may use the same topics without launching the supplied
visual executor. Start the benchmark with `observation:=rgbd executor:=none`
and run the policy in a second sourced terminal.

The built-in adapter passes simulator-matched brick meshes, vertex normals and
the exact rotational symmetries to FoundationPose. Detection poses use the
brick body centre expected by the executor. Before closing the fingers, the
executor aligns the downward tool axis over that centre; the bridge only
confirms a grasp when both finger pads contact the same upright part.
