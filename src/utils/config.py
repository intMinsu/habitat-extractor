# --- configs & profiles -------------------------------------------------------
from dataclasses import dataclass, field, replace
from typing import List, Tuple, Optional
import argparse
from src.profiles import Profile, PROFILES

@dataclass
class RenderCfg:
    """
    Rendering, I/O, and dataset selection.

    Notes
    -----
    • H/W are FINAL output sizes. The simulator renders at (H*SSAA, W*SSAA) and
      RGB is downsampled (area), depth is downsampled (min-pool).
    • Angles are in degrees; height is in centimeters.
    • `hfov_deg` applies to both RGB and Depth so intrinsics stay consistent.

    Dataset hints
    -------------
    • dataset_type: {"replicaCAD_baked","replica","hm3d_v2"}
      - replicaCAD_baked: scene_id like "sc0_00" or a *.scene_instance.json path
      - replica: scene_id like "room_0"
      - hm3d_v2: scene_id like "00800-TEEsavR23oF" or a *.glb path
    """
    # IO / random
    seed: int = 1234
    out_path: str = "export_scene"

    # Output resolution & supersampling
    H: int = 1024
    W: int = 1024
    SSAA: int = 4
    hfov_deg: float = 100.0

    # Camera pose: pitch/roll in degrees, height in cm
    pitch_start: float = -5.0
    pitch_jitter: float = 5.0
    # empty = use render.pitch_start for all rounds
    round_pitch_deg: List[float] = field(default_factory=list)
    height_start_cm: float = 150.0
    height_jitter_cm: float = 10.0
    roll_jitter: float = 3.0

    # Dataset selection
    dataset_type: str = "replicaCAD_baked"    # {"replicaCAD_baked","replica","hm3d_v2"}
    dataset_path: str = "habitat-data/replica_cad_baked_lighting"
    scene_id: str = "sc0_00"

@dataclass
class CoverageCfg:
    """
    Trajectory construction, heading policy, and sampling logistics.
    Units are meters/degrees where applicable.

    Sections
    --------
    Trajectory / path
      • traj_mode: "video" follows a smoothed path; "random" samples positions.
      • n_anchors, min_anchor_sep_m, anchor_min_clearance_m: how anchors are chosen.
      • traj_stride_m: resample spacing of the path; affects frame spacing.

    Heading policy (per-frame)
      • yaw_mode: {"tangent","target","mixed"} base heading strategy.
      • yaw_mixed_alpha: blend weight if "mixed".
      • yaw_limit_deg: pre-limit applied to the base yaw schedule along the path.
      • yaw_post_limit_deg + yaw_smooth_alpha: post-accept limit + EMA smoothing.

    Anchor-based refinement (optional)
      • anchor_yaw_enable: enable using anchors to refine headings.
      • anchor_yaw_strength (0..1): how strongly refined deltas influence the schedule.
      • anchor_yaw_max_deg: clamp per-anchor delta before interpolation (keeps turns sane).
      • Depth-openness scan (render-based):
          - anchor_viz_scan_deg / anchor_viz_scan_steps / anchor_viz_pitch_deg / anchor_viz_center_frac
          - Set anchor_viz_scan_deg = 0.0 to DISABLE the depth scan (seed yaw is kept).
      • Forward-clearance clamp (navmesh-based):
          - anchor_yaw_min_fwd_clear_m, anchor_yaw_scan_deg, anchor_yaw_scan_steps
          - Set anchor_yaw_min_fwd_clear_m = 0.0 to DISABLE the clearance clamp.

    Deterministic coverage (optional)
      • deterministic_coverage + cov_offsets_deg: build a fixed {vi,yaw} playlist.
      • cov_max_retries: small in-place yaw nudges if a planned shot fails gating.
      • cov_at_same_pos: if False, advance to the next waypoint after an accepted shot.
      • coverage_rounds: which round indices consume the playlist.

    Rounds / heights
      • two_rounds + height2_cm: perform a second pass at a different height.
      • round_yaw_offset_deg: per-round yaw biases added to the heading schedule.

    Misc view sampling
      • min_geo/max_geo: geodesic distance range for look-at targets.
      • max_rejects_per_visit: skip to the next waypoint after too many rejects.
    """
    n_views: int = 50
    max_attempts: int = 30
    max_rejects_per_visit: int = 20

    # Geodesic look-at distance when sampling targets
    min_geo: float = 1.0
    max_geo: float = 7.0

    # Trajectory
    traj_mode: str = "video"                  # {"video","random"}
    n_anchors: int = 15
    traj_stride_m: float = 0.40
    min_anchor_sep_m: float = 2.0
    two_rounds: bool = False
    height2_cm: float = 90.0
    anchor_min_clearance_m: float = 0.25

    # Heading schedule
    yaw_mode: str = "mixed"                   # {"tangent","target","mixed"}
    yaw_mixed_alpha: float = 0.35
    yaw_limit_deg: float = 25.0
    round_yaw_offset_deg: List[float] = field(default_factory=lambda: [0.0]) # [0.0], [0.0 90.0]
    yaw_post_limit_deg: float = 12.0
    yaw_smooth_alpha: float = 0.35

    # Anchor-driven heading blending
    anchor_yaw_enable: bool = False
    anchor_yaw_strength: float = 0.6
    anchor_yaw_tries: int = 24
    anchor_yaw_max_deg: float = 75.0

    # Forward-clearance refinement for anchor yaws (navmesh-based)
    # Set anchor_yaw_min_fwd_clear_m = 0.0 to DISABLE the clearance clamp.
    anchor_yaw_min_fwd_clear_m: float = 0.35
    anchor_yaw_scan_deg: float = 60.0
    anchor_yaw_scan_steps: int = 11

    # Depth-openness scan at anchors (render-based)
    # Set anchor_viz_scan_deg = 0.0 to DISABLE the depth scan (keeps the seed yaw).
    anchor_viz_scan_deg: float = 75.0
    anchor_viz_scan_steps: int = 9
    anchor_viz_pitch_deg: float = -5.0
    anchor_viz_center_frac: float = 0.28
    anchor_viz_arrow_len_m: float = 0.6
    save_anchor_images: bool = False

    # Deterministic coverage
    deterministic_coverage: bool = False
    cov_offsets_deg: List[float] = field(default_factory=lambda: [0.0, 90.0, -90.0])
    cov_at_same_pos: bool = False
    cov_max_retries: int = 4

    # which rounds should run the deterministic coverage playlist
    # Default: only round 0 (first height pass)
    coverage_rounds: Tuple[int, ...] = (0,)


@dataclass
class QualityCfg:
    """
    Per-view gating thresholds tuned for SFM-friendly frames.

    Gates (all must pass)
    ---------------------
    • min_valid_frac : minimum fraction of finite depths (avoid heavy occlusion).
    • min_rel_depth_std : normalized depth variation; discourages flat/textureless walls.
    • min_normal_disp : dispersion of normals; encourages multi-plane structure.
    • min_grad_mean : average image gradient magnitude; rejects blurry/flat images.
    • min_kpts : minimum total feature keypoints detected.
    • grid_nx/grid_ny + min_kpts_per_cell : uniformity of features across the image.
    """
    min_normal_disp: float = 0.2
    min_valid_frac: float = 0.50
    min_rel_depth_std: float = 0.10
    min_grad_mean: float = 5.0
    min_kpts: int = 100
    grid_nx: int = 8
    grid_ny: int = 8
    min_kpts_per_cell: int = 0





def _apply_profile_defaults(render: RenderCfg, cov: CoverageCfg, qual: QualityCfg, profile: str):
    if profile in (None, "", Profile.NONE):
        return render, cov, qual
    bundle = PROFILES.get(profile)
    if not bundle:
        return render, cov, qual
    r = replace(render, **bundle.get("render", {})) if "render" in bundle else render
    c = replace(cov, **bundle.get("coverage", {})) if "coverage" in bundle else cov
    q = replace(qual, **bundle.get("quality", {})) if "quality" in bundle else qual
    return r, c, q


def _parser_with_defaults(render: RenderCfg, cov: CoverageCfg, qual: QualityCfg) -> argparse.ArgumentParser:
    """
    Build an argparse parser using current dataclass values as defaults.
    This keeps CLI compatible while letting profiles change defaults cleanly.
    """
    p = argparse.ArgumentParser(
        description=("Scene exporter (RGB/Depth/Normals/Pointmap + COLMAP). "
                     "Supports ReplicaCAD baked, Replica, and HM3D v2.")
    )

    # profile (visible in --help)
    p.add_argument("--profile", type=str, default=Profile.NONE,
                   choices=[Profile.NONE] + list(PROFILES.keys()),
                   help="Preset that overrides defaults; explicit flags still win.")

    # RenderCfg
    p.add_argument("--seed", type=int, default=render.seed)
    p.add_argument("--H", type=int, default=render.H)
    p.add_argument("--W", type=int, default=render.W)
    p.add_argument("--SSAA", type=int, default=render.SSAA)
    p.add_argument("--hfov-deg", type=float, default=render.hfov_deg)
    p.add_argument("--pitch-start", type=float, default=render.pitch_start)
    p.add_argument("--round-pitch-deg", type=float, nargs="*", default=render.round_pitch_deg)
    p.add_argument("--pitch-jitter", type=float, default=render.pitch_jitter)
    p.add_argument("--height-start-cm", type=float, default=render.height_start_cm)
    p.add_argument("--height-jitter-cm", type=float, default=render.height_jitter_cm)
    p.add_argument("--roll-jitter", type=float, default=render.roll_jitter)
    p.add_argument("--dataset-type", type=str, default=render.dataset_type,
                   choices=["replicaCAD_baked", "replica", "hm3d_v2"])
    p.add_argument("--dataset-path", type=str, default=render.dataset_path)
    p.add_argument("--scene-id", type=str, default=render.scene_id)
    p.add_argument("--out-path", type=str, default=render.out_path)

    # CoverageCfg
    p.add_argument("--n-views", type=int, default=cov.n_views)
    p.add_argument("--max-attempts", type=int, default=cov.max_attempts)
    p.add_argument("--max-rejects-per-visit", type=int, default=cov.max_rejects_per_visit)
    p.add_argument("--min-geo", type=float, default=cov.min_geo)
    p.add_argument("--max-geo", type=float, default=cov.max_geo)

    p.add_argument("--traj-mode", type=str, default=cov.traj_mode, choices=["video", "random"])
    p.add_argument("--n-anchors", type=int, default=cov.n_anchors)
    p.add_argument("--traj-stride-m", type=float, default=cov.traj_stride_m)
    p.add_argument("--two-rounds", action="store_true", default=cov.two_rounds)
    p.add_argument("--height2-cm", type=float, default=cov.height2_cm)
    p.add_argument("--min-anchor-sep-m", type=float, default=cov.min_anchor_sep_m)

    p.add_argument("--yaw-mode", type=str, default=cov.yaw_mode, choices=["tangent", "target", "mixed"])
    p.add_argument("--yaw-mixed-alpha", type=float, default=cov.yaw_mixed_alpha)
    p.add_argument("--yaw-limit-deg", type=float, default=cov.yaw_limit_deg)
    p.add_argument("--round-yaw-offset-deg", type=float, nargs="*", default=cov.round_yaw_offset_deg)
    p.add_argument("--yaw-post-limit-deg", type=float, default=cov.yaw_post_limit_deg)
    p.add_argument("--yaw-smooth-alpha", type=float, default=cov.yaw_smooth_alpha)

    p.add_argument("--anchor-yaw-enable", action="store_true", default=cov.anchor_yaw_enable)
    p.add_argument("--anchor-yaw-strength", type=float, default=cov.anchor_yaw_strength)
    p.add_argument("--anchor-yaw-tries", type=int, default=cov.anchor_yaw_tries)
    p.add_argument("--anchor-yaw-max-deg", type=float, default=cov.anchor_yaw_max_deg)

    p.add_argument("--anchor-min-clearance-m", type=float, default=cov.anchor_min_clearance_m)

    p.add_argument("--anchor-yaw-min-fwd-clear-m", type=float, default=cov.anchor_yaw_min_fwd_clear_m)
    p.add_argument("--anchor-yaw-scan-deg", type=float, default=cov.anchor_yaw_scan_deg)
    p.add_argument("--anchor-yaw-scan-steps", type=int, default=cov.anchor_yaw_scan_steps)

    p.add_argument("--anchor-viz-scan-deg", type=float, default=cov.anchor_viz_scan_deg)
    p.add_argument("--anchor-viz-scan-steps", type=int, default=cov.anchor_viz_scan_steps)
    p.add_argument("--anchor-viz-pitch-deg", type=float, default=cov.anchor_viz_pitch_deg)
    p.add_argument("--anchor-viz-center-frac", type=float, default=cov.anchor_viz_center_frac)
    p.add_argument("--anchor-viz-arrow-len-m", type=float, default=cov.anchor_viz_arrow_len_m)
    p.add_argument("--save-anchor-images", action="store_true", default=cov.save_anchor_images)

    p.add_argument("--deterministic-coverage", action="store_true", default=cov.deterministic_coverage)
    p.add_argument("--cov-offsets-deg", type=float, nargs="*", default=cov.cov_offsets_deg)
    p.add_argument("--cov-at-same-pos", action="store_true", default=cov.cov_at_same_pos)
    p.add_argument("--cov-max-retries", type=int, default=cov.cov_max_retries)

    # QualityCfg
    p.add_argument("--min-normal-disp", type=float, default=qual.min_normal_disp)
    p.add_argument("--min-valid-frac", type=float, default=qual.min_valid_frac)
    p.add_argument("--min-rel-depth-std", type=float, default=qual.min_rel_depth_std)
    p.add_argument("--min-grad-mean", type=float, default=qual.min_grad_mean)
    p.add_argument("--min-kpts", type=int, default=qual.min_kpts)
    p.add_argument("--grid-nx", type=int, default=qual.grid_nx)
    p.add_argument("--grid-ny", type=int, default=qual.grid_ny)
    p.add_argument("--min-kpts-per-cell", type=int, default=qual.min_kpts_per_cell)

    return p


def parse_configs_from_cli() -> Tuple[RenderCfg, CoverageCfg, QualityCfg]:
    """
    Two-phase parse to honor --profile:
      1) read only --profile
      2) apply profile to defaults
      3) build full parser with those defaults, then parse the rest
    Explicit CLI flags always override profile/defaults.
    """
    # Phase 1: just the profile
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument("--profile", type=str, default=Profile.NONE)
    known, _ = mini.parse_known_args()

    # Start with library defaults, then profile overrides
    render, cov, qual = RenderCfg(), CoverageCfg(), QualityCfg()
    render, cov, qual = _apply_profile_defaults(render, cov, qual, known.profile)

    # Phase 2: full parse with updated defaults
    parser = _parser_with_defaults(render, cov, qual)
    args = parser.parse_args()

    # Rebuild dataclasses from the parsed args (explicit flags win)
    render = replace(render,
        seed=args.seed, out_path=args.out_path,
        H=args.H, W=args.W, SSAA=args.SSAA, hfov_deg=args.hfov_deg,
        pitch_start=args.pitch_start,
        round_pitch_deg=args.round_pitch_deg or render.round_pitch_deg,
        pitch_jitter=args.pitch_jitter,
        height_start_cm=args.height_start_cm, height_jitter_cm=args.height_jitter_cm,
        roll_jitter=args.roll_jitter,
        dataset_type=args.dataset_type, dataset_path=args.dataset_path, scene_id=args.scene_id)

    cov = replace(cov,
        n_views=args.n_views, max_attempts=args.max_attempts, max_rejects_per_visit=args.max_rejects_per_visit,
        min_geo=args.min_geo, max_geo=args.max_geo,
        traj_mode=args.traj_mode, n_anchors=args.n_anchors, traj_stride_m=args.traj_stride_m,
        min_anchor_sep_m=args.min_anchor_sep_m, two_rounds=args.two_rounds, height2_cm=args.height2_cm,
        yaw_mode=args.yaw_mode, yaw_mixed_alpha=args.yaw_mixed_alpha, yaw_limit_deg=args.yaw_limit_deg,
        round_yaw_offset_deg=args.round_yaw_offset_deg or cov.round_yaw_offset_deg,  # keep non-empty
        yaw_post_limit_deg=args.yaw_post_limit_deg, yaw_smooth_alpha=args.yaw_smooth_alpha,
        anchor_yaw_enable=args.anchor_yaw_enable, anchor_yaw_strength=args.anchor_yaw_strength,
        anchor_yaw_tries=args.anchor_yaw_tries, anchor_yaw_max_deg=args.anchor_yaw_max_deg,
        anchor_min_clearance_m=args.anchor_min_clearance_m,
        anchor_yaw_min_fwd_clear_m=args.anchor_yaw_min_fwd_clear_m,
        anchor_yaw_scan_deg=args.anchor_yaw_scan_deg, anchor_yaw_scan_steps=args.anchor_yaw_scan_steps,
        anchor_viz_scan_deg=args.anchor_viz_scan_deg, anchor_viz_scan_steps=args.anchor_viz_scan_steps,
        anchor_viz_pitch_deg=args.anchor_viz_pitch_deg, anchor_viz_center_frac=args.anchor_viz_center_frac,
        anchor_viz_arrow_len_m=args.anchor_viz_arrow_len_m, save_anchor_images=args.save_anchor_images,
        deterministic_coverage=args.deterministic_coverage,
        cov_offsets_deg=args.cov_offsets_deg or cov.cov_offsets_deg,
        cov_at_same_pos=args.cov_at_same_pos, cov_max_retries=args.cov_max_retries)

    qual = replace(qual,
        min_normal_disp=args.min_normal_disp, min_valid_frac=args.min_valid_frac,
        min_rel_depth_std=args.min_rel_depth_std, min_grad_mean=args.min_grad_mean,
        min_kpts=args.min_kpts, grid_nx=args.grid_nx, grid_ny=args.grid_ny,
        min_kpts_per_cell=args.min_kpts_per_cell)

    return render, cov, qual