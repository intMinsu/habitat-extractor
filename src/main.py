import os, json, math, argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import cv2
from collections import defaultdict

import habitat_sim
from habitat_sim.sensor import SensorType, SensorSubType, CameraSensorSpec
from habitat_sim.agent import AgentConfiguration
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn

from src.utils.config import RenderCfg, parse_configs_from_cli
from src.utils.geometry import (
    unproject_depth_to_points_cam,
    normals_from_depth,
    intrinsics_from_sensor
)
from src.utils.quaternion_helper import T_c2w_from_sensor_state, rotate_vec3_from_quat_axis
from src.utils.vis import (
    colorize_depth,
    colorize_normal,
    save_ply,
    merge_per_view_pointclouds_fast
)
from src.utils.colmap_utils import export_colmap_reconstruction
from src.utils.habitat_sample import (
    yaw_towards,
    compute_path_yaws,
    sample_target,
    is_good_view,
    wrap_pi,
    blend_yaw,
    set_agent_pose,
)
from src.utils.habitat_traj import (
    make_video_like_trajectory,
    limit_yaw_rate,
    compute_anchor_headings,
    build_yaw_schedule_from_anchors,
    plan_coverage_playlist,
)
from src.utils.habitat_traj_vis import (
    save_birdeye_poses_png,
    overlay_pose_frusta_as_points_on_pointcloud,
    write_glb_pointcloud_with_frusta
)
from src.utils.dataset_resolver import resolve_dataset_and_scene

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def scale_intrinsics(intr: dict, s: float) -> dict:
    """Scale intrinsics by factor s (e.g., from super-res -> target)."""
    out = dict(intr)
    out["W"] = int(round(intr["W"] * s))
    out["H"] = int(round(intr["H"] * s))
    out["fx"] = float(intr["fx"]) * s
    out["fy"] = float(intr["fy"]) * s
    out["cx"] = (out["W"]) / 2.0
    out["cy"] = (out["H"]) / 2.0
    out["K"] = [
        [out["fx"], 0.0, out["cx"]],
        [0.0, out["fy"], out["cy"]],
        [0.0, 0.0, 1.0],
    ]
    return out

def downsample_rgb_area(rgb_hi, W, H):
    return cv2.resize(rgb_hi, (W, H), interpolation=cv2.INTER_AREA)

def downsample_depth_minpool(depth_hi, mask_hi, SSAA, W, H):
    """
    Edge-safe downsampling: choose the *closest* valid depth per output pixel (min-pool).
    Avoids foreground/background z-bleed at discontinuities.
    """
    if SSAA <= 1:
        mask = np.isfinite(depth_hi) & (depth_hi > 0)
        depth = np.where(mask, depth_hi, np.nan)
        return depth, mask # reshape to (H, SSAA, W, SSAA)
    Hs, Ws = depth_hi.shape
    h, w = Hs // SSAA, Ws // SSAA

    depth_block = depth_hi[:h * SSAA, :w * SSAA].reshape(h, SSAA, w, SSAA)
    mask_block = mask_hi[:h * SSAA, :w * SSAA].reshape(h, SSAA, w, SSAA) # set invalid to +inf, take min along (1, 3)
    depth_block = np.where(mask_block, depth_block, np.inf)
    depth_min = depth_block.min(axis=(1, 3))

    mask = np.isfinite(depth_min) & (depth_min > 0) & (depth_min < np.inf)
    depth = np.where(mask, depth_min, np.nan) # handle possible non-divisible sizes (unlikely if W,H divisible by SSAA)

    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    return depth, mask

def render_and_gate(
    sim, agent, p_local, yaw_try, *,
    pitch, roll, dh, base_y_offset,
    W, H, SSAA, K, qual
):
    """
    Render one trial at (p_local, yaw_try) with jitter (pitch/roll/dh), then apply gates.

    Returns:
        ok, rgb_img, depth, mask, n_cam  (n_cam is camera-space normals)

    Example:
        ok, rgb, depth, mask, n_cam = render_and_gate(
            sim, agent, p, yaw_try,
            pitch=math.radians(-5), roll=0.0, dh=0.0, base_y_offset=0.0,
            W=W, H=H, SSAA=SSAA, K=K, qual=qual
        )
    """
    st = agent.get_state()
    st.position = mn.Vector3(float(p_local[0]), float(p_local[1] + base_y_offset + dh), float(p_local[2]))
    q_yaw = quat_from_angle_axis(float(yaw_try), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right = rotate_vec3_from_quat_axis((1.0, 0.0, 0.0), q_yaw)
    q_pitch = quat_from_angle_axis(float(pitch), right)
    q_yaw_pitch = q_pitch * q_yaw
    fwd = rotate_vec3_from_quat_axis((0.0, 0.0, -1.0), q_yaw_pitch)
    q_roll = quat_from_angle_axis(float(roll), fwd)
    st.rotation = q_roll * q_pitch * q_yaw
    agent.set_state(st)

    obs = sim.get_sensor_observations()
    rgb_hi = obs["rgba"][..., :3]
    depth_hi = obs["depth"].astype(np.float32)
    mask_hi = np.isfinite(depth_hi) & (depth_hi > 0)

    rgb_img = downsample_rgb_area(rgb_hi, W, H)
    depth, mask = downsample_depth_minpool(depth_hi, mask_hi, SSAA, W, H)
    n_cam = normals_from_depth(depth, K)

    ok = is_good_view(
        depth=depth, mask=mask, n_cam=n_cam, rgb=rgb_img, K=K,
        min_valid_frac=float(qual.min_valid_frac),
        min_rel_depth_std=float(qual.min_rel_depth_std),
        min_normal_disp=float(qual.min_normal_disp),
        min_grad_mean=float(qual.min_grad_mean),
        min_kpts=int(qual.min_kpts),
        grid_nx=int(qual.grid_nx), grid_ny=int(qual.grid_ny),
        min_kpts_per_cell=int(qual.min_kpts_per_cell)
    )
    return ok, rgb_img, depth, mask, n_cam


def smooth_yaw(prev_yaw_acc, yaw_now, *, yaw_post_limit_deg: float, yaw_smooth_alpha: float):
    """
    Clamp the delta from the previous *accepted* yaw, then EMA blend.

    Returns:
        yaw_final (radians)

    Example:
        yaw_sm = smooth_yaw(prev, cand, yaw_post_limit_deg=8.0, yaw_smooth_alpha=0.4)
    """
    if prev_yaw_acc is None:
        return yaw_now
    d = math.atan2(math.sin(yaw_now - prev_yaw_acc), math.cos(yaw_now - prev_yaw_acc))
    maxd = math.radians(float(yaw_post_limit_deg))
    d = float(np.clip(d, -maxd, maxd))
    yaw_limited = wrap_pi(prev_yaw_acc + d)
    a = float(yaw_smooth_alpha)
    return blend_yaw(prev_yaw_acc, yaw_limited, a) if a > 0.0 else yaw_limited


def save_outputs(out_dir: Path, idx: int, *, rgb, depth, mask, n_cam, pts_w_img, pts_w_flat, rgb_flat):
    """
    #### saving part ####
    Write all per-view artifacts to disk.

    Files:
      images/rgb_{idx}.png
      depth_vis/depth_vis_{idx}.png
      depth/depth_{idx}.exr
      normal/normal_{idx}.png
      mask/mask_{idx}.png
      points_ply/points_{idx}.ply
      points_exr/points_{idx}.exr

    Example:
        save_outputs(OUT, i, rgb=rgb_img, depth=depth, mask=mask, n_cam=n_cam,
                     pts_w_img=pts_w_img, pts_w_flat=pts_w.reshape(-1,3), rgb_flat=rgb_img.reshape(-1,3))
    """
    iio.imwrite(out_dir / "images" / f"rgb_{idx:05d}.png", rgb)
    depth_vis = colorize_depth(depth, normalize=False)
    iio.imwrite(out_dir / "depth_vis" / f"depth_vis_{idx:05d}.png", depth_vis)
    cv2.imwrite(str(out_dir / "depth" / f"depth_{idx:05d}.exr"), depth,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    normal_vis = colorize_normal(np.nan_to_num(n_cam, nan=0.0))
    iio.imwrite(out_dir / "normal" / f"normal_{idx:05d}.png", normal_vis)
    cv2.imwrite(str(out_dir / "mask" / f"mask_{idx:05d}.png"), (mask * 255).astype(np.uint8))
    save_ply(out_dir / "points_ply" / f"points_{idx:05d}.ply", pts_w_flat, rgb_flat)
    cv2.imwrite(str(out_dir / "points_exr" / f"points_{idx:05d}.exr"), pts_w_img,
                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])


def pose_meta_from_state(s, T_c2w, idx: int, round_idx: int, round_h: float) -> dict:
    """
    Pack pose metadata dict from a Habitat sensor state.

    Example:
        meta = pose_meta_from_state(s, T_c2w, idx=i, round_idx=round_idx, round_h=round_h)
    """
    try:
        px, py, pz = float(s.position.x), float(s.position.y), float(s.position.z)
    except AttributeError:
        px, py, pz = [float(v) for v in np.asarray(s.position).reshape(3)]
    rot = s.rotation
    if hasattr(rot, "vector") and hasattr(rot, "scalar"):
        rotation_xyzw = [float(rot.vector[0]), float(rot.vector[1]), float(rot.vector[2]), float(rot.scalar)]
    elif all(hasattr(rot, k) for k in ("x", "y", "z", "w")):
        rotation_xyzw = [float(rot.x), float(rot.y), float(rot.z), float(rot.w)]
    else:
        arr = np.array(rot, dtype=np.float64).ravel()
        rotation_xyzw = arr.tolist() if arr.size == 4 else [0.0, 0.0, 0.0, 1.0]
    return {
        "frame": idx,
        "round": round_idx,
        "cam_height_m": round_h,
        "T_c2w": T_c2w.tolist(),
        "position_world": [px, py, pz],
        "rotation_xyzw": rotation_xyzw,
    }


def save_reject_thumbnail(out_dir: Path, idx_acc: int, rgb_img):
    """
    Save a tiny PNG into rejected_images/ for a failed trial.

    Example:
        save_reject_thumbnail(OUT, accepted, rgb_img)
    """
    thumb = cv2.resize(rgb_img, (128, 128), interpolation=cv2.INTER_AREA)
    reject_dir = out_dir / "rejected_images"
    reject_dir.mkdir(parents=True, exist_ok=True)
    # one thumbnail per reject attempt, monotonically increasing suffix
    existing = list(reject_dir.glob(f"rgb_{idx_acc:05d}_reject_*.png"))
    next_id = 0 if not existing else (max(int(p.stem.split("_")[-1]) for p in existing) + 1)
    iio.imwrite(reject_dir / f"rgb_{idx_acc:05d}_reject_{next_id:03d}.png", thumb)

def _box_warn(lines: list):
    bar = "=" * 78
    print("\n" + bar)
    for ln in lines:
        print(f"WARNING: {ln}")
    print(bar + "\n")

def _warn_underprovisioned(
    cov,
    *,
    n_rounds: int,
    n_anchor_idxs: int,
    coverage_playlist_len: int
):
    """
    Emit warnings when the view budget can't cover anchors or the deterministic playlist.
    """
    msgs = []

    # 1) Not enough views to even touch each anchor once (rough rule of thumb)
    if cov.anchor_yaw_enable and n_anchor_idxs > 0 and cov.n_views < n_anchor_idxs:
        msgs.append(
            f"n_views={cov.n_views} < num_anchors={n_anchor_idxs}. "
            "Some anchors may never be visited; anchor-based refinement/coverage could be ineffective."
        )

    # 2) Deterministic coverage: total planned shots vs. view budget
    if cov.deterministic_coverage and coverage_playlist_len > 0 and cov.n_views < coverage_playlist_len:
        msgs.append(
            f"Deterministic coverage playlist has {coverage_playlist_len} planned shots "
            f"but n_views={cov.n_views}. Some planned views will be skipped."
        )

    # 3) Per-round pressure if coverage is restricted to specific rounds
    if cov.deterministic_coverage and hasattr(cov, "coverage_rounds"):
        valid_rounds = [r for r in cov.coverage_rounds if 0 <= int(r) < max(1, n_rounds)]
        if valid_rounds:
            shots_total = coverage_playlist_len
            # naive even split across selected rounds
            per_round_need = int(math.ceil(shots_total / max(1, len(valid_rounds))))
            per_round_budget = int(math.ceil(cov.n_views / max(1, n_rounds)))
            if per_round_budget < per_round_need:
                msgs.append(
                    f"Coverage rounds={tuple(valid_rounds)} need ~{per_round_need}/round "
                    f"but budget allows ~{per_round_budget}/round. Expect truncation."
                )

    if msgs:
        _box_warn(msgs)

# -------------------- Simulator --------------------
def make_sim(
        render: RenderCfg,
        H_SUPER: int,
        W_SUPER: int,
        CAM_HEIGHT_M: float,
        HFOV_DEG: float,
        dataset_cfg: str,
        scene_id: str
):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = dataset_cfg
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True
    sim_cfg.override_scene_light_defaults = True

    if render.dataset_type == "replica":
        sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY
    elif render.dataset_type == "hm3d_v2":
        sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY
    else:
        sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY

    # All sensors must have the same resolution and FOV!
    rgb = CameraSensorSpec()
    rgb.uuid = "rgba"
    rgb.sensor_type = SensorType.COLOR
    rgb.sensor_subtype = SensorSubType.PINHOLE
    rgb.resolution = [int(H_SUPER), int(W_SUPER)]
    rgb.hfov = float(HFOV_DEG)
    rgb.position = [0.0, float(CAM_HEIGHT_M), 0.0]
    rgb.orientation = [0.0, 0.0, 0.0]  # keep level; tilt agent each frame

    depth = CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = SensorType.DEPTH
    depth.sensor_subtype = SensorSubType.PINHOLE
    depth.resolution = [int(H_SUPER), int(W_SUPER)]
    depth.hfov = float(HFOV_DEG)  # ensure depth matches RGB FOV
    depth.position = [0.0, float(CAM_HEIGHT_M), 0.0]
    depth.orientation = [0.0, 0.0, 0.0]

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    # print(f"cfg : {cfg}")
    sim = habitat_sim.Simulator(cfg)
    agent = sim.get_agent(agent_id=0)

    # Try to auto-load a navmesh if Habitat didn't load one.
    if not sim.pathfinder.is_loaded:
        try:
            p = Path(scene_id)
            candidates = []
            # HM3D: .../xxx.basis.glb -> .../xxx.basis.navmesh (or .navmesh)
            if p.suffix.endswith("glb"):
                candidates += [p.with_suffix(".basis.navmesh"), p.with_suffix(".navmesh")]
            # Replica: scene_instance.json or mesh_*.ply -> look next to them
            if p.suffix in (".json", ".ply"):
                candidates += [
                    p.parent / "mesh_semantic.navmesh",
                    p.parent / "mesh_preseg_semantic.navmesh",
                ]
            for nav in candidates:
                if nav.exists():
                    sim.pathfinder.load_nav_mesh(str(nav))
                    break
        except Exception:
            pass

    return sim, agent, rgb, depth


# -------------------- Main --------------------
def main():
    """
    Export SFM-friendly views from a Habitat scene.

    Pipeline
    --------
    1) Path construction ("video" mode):
       anchors → geodesic tour → densify + Chaikin → resample to ~fixed stride.
    2) Heading selection per frame:
       path tangent (base) optionally refined at anchors using:
         • Depth 'openness' scan around seed
         • Forward-clearance clamp on the navmesh
       Then apply yaw-rate limit and optional EMA smoothing between accepted frames.
    3) Render & gating:
       render RGB/Depth at SSAA×, downsample (area for RGB, min-pool for depth),
       compute normals; accept frames that pass photometric/geometry gates
       (valid frac, rel depth std, normal dispersion, gradients, keypoint coverage).
    4) Outputs per accepted view:
       RGB/PNG, Depth/EXR, Normal/PNG, ValidMask/PNG, per-view PLY + per-pixel XYZ/EXR,
       and pose metadata (T_c2w, position, quaternion).
    5) Reconstruction & visualization:
       • COLMAP export: shared PINHOLE intrinsics + all poses/images
       • Point cloud fusion: merge all per-view XYZ EXRs → global sparse/fine PLYs
       • Trajectory viz:
           - PLY with frusta points overlaid
           - GLB with solid frusta, path curve, anchors, first/last markers
           - Bird’s-eye PNG of poses/anchors/path

    Key helpers
    -----------
    • compute_anchor_headings(...)      : depth-scan + forward-clear clamp at anchors
    • build_yaw_schedule_from_anchors() : interpolate/clamp anchor yaw deltas → full schedule
    • plan_coverage_playlist(...)       : deterministic per-anchor shots (best ± offsets)

    Quick start (Replica)
    ---------------------
    python scripts/habitat/extractor.py \\
      --profile sfm_hq \\
      --n_anchors=18 \\
      --n_views=200 \\
      --dataset-type replica \\
      --dataset-path habitat-data/replica \\
      --scene-id room_0 \\
      --out-path export-replica/room_0

    Notes
    -----
    • Profiles provide sane defaults; any CLI flag can override profile values.
    • Deterministic coverage is off by default (broad, diverse sampling). Enable it
      when you want repeatable viewpoints or view-sphere coverage.
    """
    render, cov, qual = parse_configs_from_cli()
    rng = np.random.default_rng(render.seed)

    # Resolve dataset + scene
    DATASET_CFG, SCENE_ID = resolve_dataset_and_scene(argparse.Namespace(
        dataset_type=render.dataset_type,
        dataset_path=render.dataset_path,
        scene_id=render.scene_id,
        out_path=render.out_path
    ))
    print(f"[load] dataset_cfg = {DATASET_CFG}")
    print(f"[load] scene_id    = {SCENE_ID}")

    # Output folders
    OUT = Path(render.out_path)
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["images", "depth_vis", "depth", "normal", "mask", "points_ply", "points_exr", "rejected_images",
              "traj_vis", "global_points_ply"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    # Supersampling & FOV
    H, W = int(render.H), int(render.W)
    SSAA = max(1, int(render.SSAA))
    H_SUPER, W_SUPER = H * SSAA, W * SSAA

    HFOV_DEG = float(np.clip(render.hfov_deg, 30.0, 120.0))

    # Pitch / height (radians/meters)
    CAM_PITCH = math.radians(float(render.pitch_start))
    PITCH_JITTER_STD = math.radians(float(render.pitch_jitter))
    CAM_HEIGHT_M = float(render.height_start_cm) / 100.0
    HEIGHT_JITTER_STD_M = float(render.height_jitter_cm) / 100.0
    ROLL_JITTER_STD = math.radians(float(render.roll_jitter))

    # Simulator
    sim, agent, rgb_spec, depth_spec = make_sim(
        render, H_SUPER, W_SUPER, CAM_HEIGHT_M, HFOV_DEG, DATASET_CFG, SCENE_ID
    )
    sim.pathfinder.seed(render.seed)

    # Intrinsics at super-res → scale to output res
    intr_super = intrinsics_from_sensor(sim, rgb_uuid="rgba", hfov_deg_hint=HFOV_DEG,
                                        fallback_resolution=(H_SUPER, W_SUPER))
    intr = scale_intrinsics(intr_super, 1.0 / SSAA)
    with open(OUT / "camera_intrinsics.json", "w") as f:
        json.dump(intr, f, indent=2)
    K = np.array(intr["K"], dtype=np.float32)

    poses_meta = []
    accepted = 0
    rej_counters = defaultdict(int)  # key = accepted index, val = #rejects so far
    TOTAL_MAX_TRIES = int(cov.n_views) * int(cov.max_attempts)
    tries = 0

    #### base trajectory by selecting informative anchors START ####
    """
    1. Only make_video_like_trajectory + tangent yaws: fastest, good for quick fly-throughs or very open scenes.
    --- optional refinement (anchor_yaw_enable, anchor_yaw_strength>0) ---
    2. Add compute_anchor_headings: indoor scenes with frequent occluders; yields more informative, stable looks.
    3. Add build_yaw_schedule_from_anchors: always recommended once you refine anchors; it spreads local decisions smoothly.
    4. Add plan_coverage_playlist: benchmarks, ablations, or debugging where you need repeatable specific views.
    5. That’s the whole story: positions → anchor-wise smart headings → smooth global schedule → (optionally) deterministic shots.
    """
    if cov.traj_mode == "video":
        base_traj, anchors, _polys = make_video_like_trajectory(
            sim,
            n_anchors=int(cov.n_anchors),
            stride_m=float(cov.traj_stride_m),
            seed=int(render.seed),
            min_sep_geo=float(cov.min_anchor_sep_m),
            closed_loop=True,
            densify_max_seg_m=0.20,
            smooth_passes=1,
            return_debug=True,
            anchor_min_clearance_m=float(cov.anchor_min_clearance_m),
        )
        base_yaws = compute_path_yaws(base_traj)

        # Prevent huge yaw change per step
        if cov.yaw_limit_deg > 0:
            base_yaws = limit_yaw_rate(base_yaws, max_rate_deg=float(cov.yaw_limit_deg))

        # Default: path-tangent only
        yaw_sched = base_yaws
        coverage_playlist = []

        # Optional: anchor-driven refinement
        if (
                base_traj.shape[0] > 0
                and len(anchors) > 0
                and cov.anchor_yaw_enable
                and float(cov.anchor_yaw_strength) > 0.0
        ):
            # Refine headings of anchors using Depth openness & Forward clearance
            # seed_yaws = the subset of base_yaws at anchor_idxs
            # best_yaws = the refined versions (depth-openness scan + forward-clearance clamp)
            anchor_idxs, seed_yaws, best_yaws, local_clearances = compute_anchor_headings(
                sim, agent,
                base_traj=base_traj,
                anchors_list=anchors,
                base_yaws=base_yaws,
                pitch_deg=float(cov.anchor_viz_pitch_deg),
                viz_scan_deg=float(cov.anchor_viz_scan_deg),
                viz_scan_steps=int(cov.anchor_viz_scan_steps),
                center_frac=float(cov.anchor_viz_center_frac),
                min_fwd_clear_m=float(cov.anchor_yaw_min_fwd_clear_m),
                yaw_scan_deg=float(cov.anchor_yaw_scan_deg),
                yaw_scan_steps=int(cov.anchor_yaw_scan_steps),
                cam_height_start_m=float(render.height_start_cm) / 100.0,
            )

            # Actual smooth yaw schedule from refined anchors(compute_anchor_headings).
            # even if the depth/clearance logic suggests a big turn,
            # we limit the injected delta to ±anchor_yaw_max_deg
            # before we interpolate over the whole path.
            yaw_sched = build_yaw_schedule_from_anchors(
                base_yaws=base_yaws,
                anchor_idxs=anchor_idxs,
                best_yaws=best_yaws,
                strength=float(cov.anchor_yaw_strength),
                max_delta_deg=float(cov.anchor_yaw_max_deg),
                yaw_limit_deg=float(cov.yaw_limit_deg),
            )

            # Optional deterministic coverage (e.g. cov_offsets_deg=[0] to save 'best_yaws' whenever)
            if cov.deterministic_coverage:
                coverage_playlist = plan_coverage_playlist(
                    anchor_idxs=anchor_idxs,
                    best_yaws=best_yaws,
                    cov_offsets_deg=cov.cov_offsets_deg,
                )

            cov_ptr = 0

            # Optional per-anchor debug snapshots (only for first round)
            if cov.save_anchor_images:
                dbg_dir = OUT / "anchor_debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                H1 = float(render.height_start_cm) / 100.0
                pitch_rad = math.radians(float(cov.anchor_viz_pitch_deg))
                for k, idx in enumerate(anchor_idxs):
                    p = base_traj[idx]
                    for tag, yaw_use in [("seed", float(seed_yaws[k])), ("best", float(best_yaws[k]))]:
                        st_save = agent.get_state()
                        set_agent_pose(
                            agent, p, yaw_use,
                            pitch_rad,  # small downward pitch for a scene-forward snapshot
                            base_cam_height_m=H1,
                            cam_height_m=H1,
                        )
                        obs = sim.get_sensor_observations()
                        rgb_hi = obs["rgba"][..., :3]
                        rgb_dbg = downsample_rgb_area(rgb_hi, W, H)
                        iio.imwrite(dbg_dir / f"anchor_{k:02d}_{tag}.png", rgb_dbg)
                        agent.set_state(st_save)
    else:
        base_traj = np.zeros((0, 3), np.float32)
        yaw_sched = np.zeros((0,), np.float32)
        coverage_playlist = []
        anchors = []
    #### base trajectory by selecting informative anchors END ####

    # -------- round planning --------
    """
    We may do multiple passes (“rounds”) over the same visit path, optionally
    at a different camera height and with a per-round yaw bias.

    round_heights:
      List of camera heights (meters), typically one or two rounds.
      Example: H1 = 1.50m, H2 = 1.20m to mix adult/child or tripod heights.

    round_yaw_offsets:
      List of per-round yaw biases (DEGREES). Each value is added to every
      frame’s yaw in that round (then wrapped to [-pi, pi)).
      Examples:
        [0, 180]  → second pass looks backward (complements coverage).
        [0, 30]   → slight azimuth sweep for richer parallax.
        [0, -20, 20] → three passes with small spreads.
      If fewer offsets than rounds are given, pad with the last value.

    target_per_round:
      Soft cap on accepted frames per round so total ≈ n_views across rounds.
      
    Example:
    1. Two heights, same look direction
    two_rounds=True, height2_cm=120, round_yaw_offset_deg=[0]
    
    2. Two passes, second is back-facing
    two_rounds=True, round_yaw_offset_deg=[0, 180]
    
    3. Single pass, slight global twist
    two_rounds=False, round_yaw_offset_deg=[15]
    """

    H1 = float(render.height_start_cm) / 100.0
    H2 = float(cov.height2_cm) / 100.0
    round_heights = [H1] + ([H2] if cov.two_rounds else [])

    # We prefer real anchor_idxs (from compute_anchor_headings); fallback to len(anchors)
    num_anchor_idxs = int(len(anchor_idxs)) if "anchor_idxs" in locals() else int(len(anchors) or 0)

    coverage_playlist_len = int(len(coverage_playlist)) if "coverage_playlist" in locals() else 0

    n_rounds = 2 if cov.two_rounds else 1
    _warn_underprovisioned(
        cov,
        n_rounds=n_rounds,
        n_anchor_idxs=num_anchor_idxs,
        coverage_playlist_len=coverage_playlist_len
    )

    round_yaw_offsets = [math.radians(float(v)) for v in (cov.round_yaw_offset_deg or [0.0])]
    if len(round_yaw_offsets) < max(1, len(round_heights)):
        round_yaw_offsets += [round_yaw_offsets[-1]] * (len(round_heights) - len(round_yaw_offsets))
    # (If round_yaw_offsets has more entries than rounds, extras are unused.)

    target_per_round = max(1, int(math.ceil(cov.n_views / max(1, len(round_heights)))))

    round_idx = 0
    for round_h in round_heights:
        if accepted >= int(cov.n_views):
            break

        # -------- visit planning (per round) --------
        base_y_offset = (round_h - float(render.height_start_cm) / 100.0)
        visit = base_traj if (cov.traj_mode == "video" and base_traj.shape[0] > 0) else None
        vi = 0
        rej_at_vi = 0
        prev_yaw_acc = None
        accepted_this_round = 0

        while accepted < int(cov.n_views) and accepted_this_round < target_per_round:
            tries += 1
            if tries > TOTAL_MAX_TRIES:
                break

            # -------- viewpoint selection (coverage or normal) --------
            """
            cov_ptr = the pointer (index) into your coverage_playlist.
            
            Step 0: choose a base position `p` and a seed yaw `yaw_base`.
              • If we have a resampled path (`visit`), use visit[vi] and the scheduled heading yaw_sched[vi].
              • Otherwise (no path), pick a random navigable point and optionally a look-at target to seed yaw.

            Step 1: decide if we are in deterministic coverage mode.
              use_coverage = (
                  cov.deterministic_coverage            # playlist mode enabled
                  and cov.traj_mode == "video"          # requires a path
                  and base_traj.shape[0] > 0
                  and cov_ptr < len(coverage_playlist)  # still have shots to consume
                  and (round_idx in getattr(cov, "coverage_rounds", []))  # coverage applies in this round
              )

            ── Coverage branch (strict & repeatable) ───────────────────────────────────────
            • Use the planned anchor index and playlist yaw:
                  plan = coverage_playlist[cov_ptr]
                  p = base_traj[plan["vi"]]
                  yaw_plan = wrap_pi(plan["yaw"] + round_yaw_offsets[round_idx])
            • Render with fixed pitch/roll/height (no jitter). If gating fails, try small yaw nudges (±5°, ±10°) up to
              `cov.cov_max_retries`. On accept: record, advance cov_ptr, and (optionally) advance `vi` if `cov.cov_at_same_pos=False`.
            • NOTE: `yaw_mode` and smoothing are intentionally bypassed for determinism here.

            ── Normal branch (scene-aware and smooth) ──────────────────────────────────────
            • Optionally pick a look-at target `q_tar` and compute `yaw_target`.
            • Choose heading by policy:
                  tangent  → yaw_final = yaw_base
                  target   → yaw_final = yaw_target
                  mixed    → yaw_final = blend_yaw(yaw_base, yaw_target, cov.yaw_mixed_alpha)
            • Apply per-round static offset: yaw_final = wrap_pi(yaw_final + round_yaw_offsets[round_idx])
            • Smooth between accepted frames:
                  yaw_final = smooth_yaw(prev_yaw_acc, yaw_final,
                                         yaw_post_limit_deg=cov.yaw_post_limit_deg,
                                         yaw_smooth_alpha=cov.yaw_smooth_alpha)
            • Add mild pose jitter (pitch/roll/height), render, and gate. On reject, save a thumbnail and possibly skip
              to the next waypoint after `cov.max_rejects_per_visit` failures. On accept, update `prev_yaw_acc` and advance.

            Result: `yaw_final` and `p` define the next candidate view; if it passes quality gates, we save outputs + pose.
            """
            # Choose base position + seed yaw
            if visit is None:
                p = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32).reshape(3)
                if not np.isfinite(p).all():
                    continue
                q_tar = sample_target(sim, p, min_geo=float(cov.min_geo), max_geo=float(cov.max_geo), tries=50)
                yaw_base = yaw_towards(p, q_tar) if q_tar is not None else rng.uniform(-math.pi, math.pi)
            else:
                if vi >= visit.shape[0]:
                    vi = 0
                p = visit[vi]
                yaw_base = float(yaw_sched[vi])

            use_coverage = (
                    cov.deterministic_coverage
                    and cov.traj_mode == "video"
                    and base_traj.shape[0] > 0
                    and cov_ptr < len(coverage_playlist)
                    and (round_idx in getattr(cov, "coverage_rounds", []))
            )

            if use_coverage:
                plan = coverage_playlist[cov_ptr]
                p = base_traj[plan["vi"]]
                yaw_plan = wrap_pi(plan["yaw"] + round_yaw_offsets[round_idx])
                pitch, roll, dh = CAM_PITCH, 0.0, 0.0

                # -------- render + gate (coverage) --------
                ok, rgb_img, depth, mask, n_cam = render_and_gate(
                    sim, agent, p, yaw_plan,
                    pitch=pitch, roll=roll, dh=dh, base_y_offset=base_y_offset,
                    W=W, H=H, SSAA=SSAA, K=K, qual=qual
                )
                y_chosen = yaw_plan

                if not ok and cov.cov_max_retries > 0:
                    for ddeg in [5.0, -5.0, 10.0, -10.0][:int(cov.cov_max_retries)]:
                        y_try = wrap_pi(yaw_plan + math.radians(ddeg))
                        ok, rgb_img, depth, mask, n_cam = render_and_gate(
                            sim, agent, p, y_try,
                            pitch=pitch, roll=roll, dh=dh, base_y_offset=base_y_offset,
                            W=W, H=H, SSAA=SSAA, K=K, qual=qual
                        )
                        if ok:
                            y_chosen = y_try
                            break

                if not ok:
                    # -------- reject handling (coverage) --------
                    save_reject_thumbnail(OUT, accepted, rgb_img)
                    cov_ptr += 1
                    continue

                yaw_final = y_chosen
                prev_yaw_acc = yaw_final
                if not cov.cov_at_same_pos and visit is not None:
                    vi = (plan["vi"] + 1) % visit.shape[0]
                cov_ptr += 1

            else:
                # Normal, possibly target/mixed + smoothing
                yaw_target = None
                if cov.yaw_mode in ("target", "mixed"):
                    q_tar = sample_target(sim, p, min_geo=float(cov.min_geo), max_geo=float(cov.max_geo), tries=30)
                    if q_tar is not None:
                        yaw_target = yaw_towards(p, q_tar)

                if cov.yaw_mode == "tangent" or yaw_target is None:
                    yaw_final = yaw_base
                elif cov.yaw_mode == "target":
                    yaw_final = yaw_target
                else:
                    yaw_final = blend_yaw(yaw_base, yaw_target, float(cov.yaw_mixed_alpha))

                yaw_final = wrap_pi(yaw_final + round_yaw_offsets[round_idx])
                yaw_final = smooth_yaw(
                    prev_yaw_acc, yaw_final,
                    yaw_post_limit_deg=float(cov.yaw_post_limit_deg),
                    yaw_smooth_alpha=float(cov.yaw_smooth_alpha)
                )

                pitch = CAM_PITCH + rng.normal(0.0, PITCH_JITTER_STD)
                roll = rng.normal(0.0, ROLL_JITTER_STD)
                dh = rng.normal(0.0, HEIGHT_JITTER_STD_M)

                # -------- render + gate (normal) --------
                ok, rgb_img, depth, mask, n_cam = render_and_gate(
                    sim, agent, p, yaw_final,
                    pitch=pitch, roll=roll, dh=dh, base_y_offset=base_y_offset,
                    W=W, H=H, SSAA=SSAA, K=K, qual=qual
                )

                if not ok:
                    # -------- reject handling (normal) --------
                    save_reject_thumbnail(OUT, accepted, rgb_img)
                    rej_at_vi += 1
                    if visit is not None and rej_at_vi >= int(cov.max_rejects_per_visit):
                        print(f"[visit] too many rejects at vi={vi} → skipping to next waypoint")
                        vi = (vi + 1) % visit.shape[0]
                        rej_at_vi = 0
                    continue

                vi += 1
                rej_at_vi = 0
                prev_yaw_acc = yaw_final

            # -------- pose meta (after ACCEPT) --------
            s = agent.get_state().sensor_states["rgba"]
            T_c2w = T_c2w_from_sensor_state(s)

            # -------- products (XYZ etc.) --------
            n_cam_nn = np.nan_to_num(n_cam, nan=0.0)
            pts_cam = unproject_depth_to_points_cam(depth, K)
            pts_w = (pts_cam @ T_c2w[:3, :3].T) + T_c2w[:3, 3]
            pts_w_img = pts_w.reshape(H, W, 3).astype(np.float32)

            # -------- saving part --------
            i = accepted
            save_outputs(
                OUT, i,
                rgb=rgb_img, depth=depth, mask=mask, n_cam=n_cam_nn,
                pts_w_img=pts_w_img,
                pts_w_flat=pts_w.reshape(-1, 3),
                rgb_flat=rgb_img.reshape(-1, 3)
            )

            # -------- record metadata --------
            poses_meta.append(pose_meta_from_state(s, T_c2w, idx=i, round_idx=round_idx, round_h=round_h))
            if accepted > 1:
                last = np.asarray(poses_meta[-2]["position_world"])
                cur = np.asarray(poses_meta[-1]["position_world"])
                print(f"[stride] accepted step = {np.linalg.norm(cur - last):.3f} m")

            accepted += 1
            accepted_this_round += 1

        # -------- post round --------
        round_idx += 1
    # Hereafter extraction is finished!!

    # -------- persist pose list --------
    with open(OUT / "poses_c2w.json", "w") as f:
        json.dump(poses_meta, f, indent=2)

    # -------- COLMAP export --------
    export_colmap_reconstruction(
        colmap_dir=OUT / "sparse",
        intr=intr,
        poses_meta=poses_meta,
        image_name_fmt="rgb_{:05d}.png",
        camera_model="PINHOLE",
        shared_camera=True,
    )

    # -------- global point-cloud fusion --------
    (OUT / "global_points_ply").mkdir(parents=True, exist_ok=True)

    # Coarser (fast) fused cloud
    merge_per_view_pointclouds_fast(
        exr_dir=OUT / "points_exr",
        rgb_dir=OUT / "images",
        out_ply=OUT / "global_points_ply" / "global_point_cloud_sparse.ply",
        voxel_size=0.02,  # ~2 cm voxels
        min_pts_per_voxel=3,
        device="cuda",  # "cpu" if no GPU
        method="packed",  # fastest
    )

    # Higher-density fused cloud
    merge_per_view_pointclouds_fast(
        exr_dir=OUT / "points_exr",
        rgb_dir=OUT / "images",
        out_ply=OUT / "global_points_ply" / "global_point_cloud_fine.ply",
        voxel_size=0.01,  # ~1 cm voxels
        min_pts_per_voxel=3,
        device="cuda",
        method="packed",
    )

    # -------- trajectory visualization --------
    every = max(1, len(poses_meta) // 200)  # thin frusta if many frames
    anchors_np = (
        np.asarray(anchors, np.float32)
        if isinstance(anchors, (list, tuple)) and len(anchors) else
        (base_traj[:1] if base_traj.shape[0] else np.zeros((0, 3), np.float32))
    )

    # PLY overlay: point-only frusta markers on the fused cloud
    overlay_pose_frusta_as_points_on_pointcloud(
        in_ply=OUT / "global_points_ply" / "global_point_cloud_fine.ply",
        out_ply=OUT / "traj_vis" / "global_point_cloud_with_pose_points.ply",
        poses_meta=poses_meta,
        intr=intr,
        near=0.06,
        far=0.22,
        edge_samples=10,
        every=every,
        add_axes=True,
        axis_len=0.10,
        frustum_color=(0, 0, 0),
        ascii=False,
    )

    # GLB overlay: wire frusta + textured near-plane thumbnails
    write_glb_pointcloud_with_frusta(
        in_ply=OUT / "global_points_ply" / "global_point_cloud_fine.ply",
        out_glb=OUT / "traj_vis" / "global_point_cloud_with_frusta.glb",
        poses_meta=poses_meta,
        intr=intr,
        near=0.03,
        far=0.15,
        every=every,  # subsample frusta if many frames
        frustum_style="wire",  # <-- edges only
        frustum_edge_radius=0.003,
        frustum_edge_sections=12,
        image_plane="near",  # <-- textured near plane
        image_dir=OUT / "images",
        image_name_fmt="rgb_{:05d}.png",
        image_facing="in", # "out"|"in"|"both" -> I don't know why "both" fails - just fix "in"
        texture_every=max(1, every),  # texture at same thinning as frusta
        max_textured=120,  # cap for file size safety
        point_subsample=1,
        visit=base_traj if base_traj.shape[0] else None,
        path_every=1,
        path_radius=0.015,
        path_color=(30, 110, 255, 255),
        anchors=anchors_np,
        anchor_radius=0.05,
        anchor_color=(255, 20, 20, 255),
        mark_first_last=True,
        first_color=(30, 220, 30, 255),
        last_color=(255, 30, 255, 255),
        marker_radius=0.07,
    )

    # Bird’s-eye PNG
    save_birdeye_poses_png(
        out_png=OUT / "traj_vis" / "poses_birdeye.png",
        poses_meta=poses_meta,
        anchors=anchors_np,
        visit=base_traj if base_traj.shape[0] else np.zeros((0, 3), np.float32),
        px=1280,
        frustum_len_m=0.45,
        frustum_width_m=0.28,
        every=every,
    )

    sim.close()


if __name__ == "__main__":
    main()
