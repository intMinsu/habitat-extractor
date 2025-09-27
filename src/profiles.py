# --- profiles: set sensible bundles of defaults --------------------------------
class Profile:
    NONE = "none"
    STANDARD = "standard"
    FAST_DEBUG = "fast_debug"
    APT_MEDIUM = "apt_medium"
    BUILDING_LARGE = "building_large"
    WAREHOUSE_XL = "warehouse_xl"

PROFILES = {
    Profile.STANDARD: dict(
        render=dict(
            H=1024, W=1024, SSAA=4, hfov_deg=95.0, pitch_start=-5, height_start_cm=150,
        ),
        coverage=dict(
            n_views=200,
            traj_mode="video",
            n_anchors=18,
            traj_stride_m=0.40,
            min_anchor_sep_m=2.0,
            two_rounds=True,
            height2_cm=90.0,
            round_yaw_offset_deg=[0.0, 15.0],
            yaw_mode="mixed",
            yaw_mixed_alpha=0.35,
            yaw_limit_deg=25.0,
            yaw_post_limit_deg=12.0,
            yaw_smooth_alpha=0.35,
            anchor_yaw_enable=True,
            anchor_yaw_strength=0.6,
            anchor_yaw_max_deg=75.0,
            anchor_min_clearance_m=0.35,
            anchor_yaw_min_fwd_clear_m=0.35,
            anchor_yaw_scan_deg=60.0,
            anchor_yaw_scan_steps=11,
            deterministic_coverage=True,
            cov_offsets_deg=[0, 20, -20], # [0] to retrieve 'best', Too large value retrieves unnatural I guess
            cov_max_retries=4,
            max_attempts=30,
            max_rejects_per_visit=20,
            coverage_rounds=(0,1) # Two rounds go deterministic
        ),
        quality=dict(
            min_valid_frac=0.50,
            min_rel_depth_std=0.10,
            min_normal_disp=0.2,
            min_grad_mean=5.0,
            min_kpts=100,
            grid_nx=8, grid_ny=8,
            min_kpts_per_cell=0,
        ),
    ),

    Profile.FAST_DEBUG: dict(
        render=dict(
            H=480, W=640, SSAA=1, hfov_deg=95.0,  pitch_start=-5, height_start_cm=150,
        ),
        coverage=dict(
            n_views=60,
            traj_mode="video",
            n_anchors=8,
            traj_stride_m=0.50,
            min_anchor_sep_m=1.5,
            two_rounds=False,
            deterministic_coverage=False,
            anchor_yaw_enable=False,
            max_attempts=8,
            max_rejects_per_visit=6,
        ),
        quality=dict(
            min_valid_frac=0.40,
            min_rel_depth_std=0.08,
            min_normal_disp=0.15,
            min_grad_mean=4.0,
            min_kpts=60,
            grid_nx=6, grid_ny=6,
            min_kpts_per_cell=0,
        ),
    ),

    # Typical multi-room apartment
    Profile.APT_MEDIUM: dict(
        render=dict(H=1080, W=1440, SSAA=4, hfov_deg=95.0, pitch_start=-5, height_start_cm=150,),
        coverage=dict(
            n_views=300,
            traj_mode="video",
            n_anchors=24,
            traj_stride_m=0.40,
            min_anchor_sep_m=1.8,
            two_rounds=True,
            height2_cm=110.0,
            round_yaw_offset_deg=[0.0, 20.0],
            yaw_mode="mixed",
            yaw_mixed_alpha=0.35,
            yaw_limit_deg=24.0,
            yaw_post_limit_deg=12.0,
            yaw_smooth_alpha=0.35,
            anchor_yaw_enable=True,
            anchor_yaw_strength=0.65,
            anchor_yaw_max_deg=70.0,
            anchor_min_clearance_m=0.30,
            anchor_yaw_min_fwd_clear_m=0.35,
            anchor_yaw_scan_deg=60.0,
            anchor_yaw_scan_steps=11,
            deterministic_coverage=True,
            cov_offsets_deg=[0, 15, -15],
            cov_max_retries=4,
            max_attempts=26,
            max_rejects_per_visit=16,
            coverage_rounds=(0,)  # only first round consumes coverage
        ),
        quality=dict(
            min_valid_frac=0.50,
            min_rel_depth_std=0.10,
            min_normal_disp=0.2,
            min_grad_mean=5.0,
            min_kpts=100,
            grid_nx=8, grid_ny=8,
            min_kpts_per_cell=0,
        ),
    ),

    # Large building floor (long corridors, many rooms)
    Profile.BUILDING_LARGE: dict(
        render=dict(H=1200, W=1600, SSAA=4, hfov_deg=95.0, pitch_start=-5, height_start_cm=150,),
        coverage=dict(
            n_views=350,
            traj_mode="video",
            n_anchors=28,
            traj_stride_m=0.50,
            min_anchor_sep_m=2.2,
            two_rounds=True,
            height2_cm=90.0,
            round_yaw_offset_deg=[0.0, 25.0],
            yaw_mode="mixed",
            yaw_mixed_alpha=0.35,
            yaw_limit_deg=25.0,
            yaw_post_limit_deg=12.0,
            yaw_smooth_alpha=0.35,
            anchor_yaw_enable=True,
            anchor_yaw_strength=0.6,
            anchor_yaw_max_deg=75.0,
            anchor_min_clearance_m=0.35,
            anchor_yaw_min_fwd_clear_m=0.35,
            anchor_yaw_scan_deg=60.0,
            anchor_yaw_scan_steps=11,
            deterministic_coverage=True,
            cov_offsets_deg=[0, 20, -20],
            cov_max_retries=4,
            max_attempts=36,
            max_rejects_per_visit=24,
            coverage_rounds=(0, 1),
        ),
        quality=dict(
            min_valid_frac=0.52,
            min_rel_depth_std=0.10,
            min_normal_disp=0.2,
            min_grad_mean=5.0,
            min_kpts=110,
            grid_nx=8, grid_ny=8,
            min_kpts_per_cell=0,
        ),
    ),

    # Huge open spaces / warehouse (long sight lines, fewer obstructions)
    Profile.WAREHOUSE_XL: dict(
        render=dict(H=900, W=1600, SSAA=4, hfov_deg=95.0, pitch_start=-5, height_start_cm=150,),
        coverage=dict(
            n_views=400,
            traj_mode="video",
            n_anchors=32,
            traj_stride_m=0.75,
            min_anchor_sep_m=3.0,
            two_rounds=False,
            yaw_mode="tangent",          # target/mixed less necessary in wide-open space
            yaw_limit_deg=20.0,
            yaw_post_limit_deg=10.0,
            yaw_smooth_alpha=0.30,
            anchor_yaw_enable=False,     # usually not needed
            deterministic_coverage=False,
            max_attempts=24,
            max_rejects_per_visit=12,
        ),
        quality=dict(
            min_valid_frac=0.45,
            min_rel_depth_std=0.08,
            min_normal_disp=0.16,
            min_grad_mean=4.0,
            min_kpts=80,
            grid_nx=6, grid_ny=6,
            min_kpts_per_cell=0,
        ),
    ),
}

# PROFILES.update({
# # As your config ...
#     Profile.CUSTOM: dict(
#         render=dict(),
#         coverage=dict(),
#         quality=dict(),
#     ),
# })