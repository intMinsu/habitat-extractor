# --- profiles: set sensible bundles of defaults --------------------------------
class Profile:
    NONE = "none"
    SMALL_ROOM_DENSE_1ROUND = "small_room_dense_1round"
    SMALL_ROOM_DENSE_2ROUND = "small_room_dense_2round"
    MULTI_ROOMS_DENSE_1ROUND = "multi_rooms_dense_1round"
    MULTI_ROOMS_DENSE_2ROUND = "multi_rooms_dense_2round"
    FAST_DEBUG = "fast_debug"

PROFILES = {
    Profile.SMALL_ROOM_DENSE_1ROUND: dict(
        render=dict(
            H=1024, W=1024, SSAA=4, hfov_deg=95.0,
            pitch_start=-5, height_start_cm=150,
            round_pitch_deg=[-5.0],
        ),
        coverage=dict(
            # Dense, single-round capture tuned for small rooms
            n_views=200,
            traj_mode="video",
            n_anchors=16,
            traj_stride_m=0.40,
            min_anchor_sep_m=1.20,  # auto-tune will refine
            two_rounds=False,  # single pass
            round_yaw_offset_deg=[0.0],

            # Heading policy
            yaw_mode="mixed",
            yaw_mixed_alpha=0.35,
            yaw_limit_deg=24.0,
            yaw_post_limit_deg=12.0,
            yaw_smooth_alpha=0.35,

            # Anchor-driven refinement
            anchor_yaw_enable=True,
            anchor_yaw_strength=0.60,
            anchor_yaw_max_deg=70.0,
            anchor_min_clearance_m=0.25,
            anchor_yaw_min_fwd_clear_m=0.30,
            anchor_yaw_scan_deg=50.0,
            anchor_yaw_scan_steps=9,

            # Deterministic coverage (offset-major; avoid pure spins in main)
            deterministic_coverage=True,
            cov_offsets_deg=[20.0, -20.0, 160.0, -160.0],
            cov_max_retries=4,
            coverage_rounds=(0,),  # only round 0 consumes playlist

            # Attempts / rejects
            max_attempts=30,
            max_rejects_per_visit=16,
        ),
        quality=dict(
            # Keep default SFM-friendly gates
            min_valid_frac=0.50,
            min_rel_depth_std=0.10,
            min_normal_disp=0.20,
            min_grad_mean=5.0,
            min_kpts=100,
            grid_nx=8, grid_ny=8,
            min_kpts_per_cell=0,
        ),
    ),

Profile.SMALL_ROOM_DENSE_2ROUND: dict(
        render=dict(
            H=1024, W=1024, SSAA=4, hfov_deg=95.0,
            pitch_start=-5, height_start_cm=150,
            # Per-round base pitch (deg): round 0, round 1
            round_pitch_deg=[-5.0, -10.0],
        ),
        coverage=dict(
            # Dense, two-round capture tuned for small rooms
            n_views=200,
            traj_mode="video",
            n_anchors=16,
            traj_stride_m=0.40,
            min_anchor_sep_m=1.20,      # auto-tune will refine
            two_rounds=True,             # two passes
            height2_cm=120.0,            # second-round camera height
            round_yaw_offset_deg=[0.0, 30.0],  # slight azimuth sweep on round 2

            # Heading policy
            yaw_mode="mixed",
            yaw_mixed_alpha=0.35,
            yaw_limit_deg=24.0,
            yaw_post_limit_deg=12.0,
            yaw_smooth_alpha=0.35,

            # Anchor-driven refinement
            anchor_yaw_enable=True,
            anchor_yaw_strength=0.60,
            anchor_yaw_max_deg=70.0,
            anchor_min_clearance_m=0.25,
            anchor_yaw_min_fwd_clear_m=0.30,
            anchor_yaw_scan_deg=50.0,
            anchor_yaw_scan_steps=9,

            # Deterministic coverage (offset-major; avoids pure spins in main)
            deterministic_coverage=True,
            cov_offsets_deg=[20.0, -20.0, 160.0, -160.0],
            cov_max_retries=4,
            coverage_rounds=(0, 1),     # consume playlist in both rounds

            # Attempts / rejects
            max_attempts=36,
            max_rejects_per_visit=18,
        ),
        quality=dict(
            min_valid_frac=0.50,
            min_rel_depth_std=0.10,
            min_normal_disp=0.20,
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

    Profile.MULTI_ROOMS_DENSE_1ROUND: dict(
            render=dict(
                H=1024, W=1024, SSAA=4, hfov_deg=95.0,
                pitch_start=-5, height_start_cm=150,
                round_pitch_deg=[-5.0],
            ),
            coverage=dict(
                # Denser coverage across several rooms; single round
                n_views=350,
                traj_mode="video",
                n_anchors=24,
                traj_stride_m=0.45,
                min_anchor_sep_m=1.60,          # auto-tune refines per scene
                two_rounds=False,
                round_yaw_offset_deg=[0.0],
                # Heading policy
                yaw_mode="mixed",
                yaw_mixed_alpha=0.35,
                yaw_limit_deg=24.0,
                yaw_post_limit_deg=12.0,
                yaw_smooth_alpha=0.35,
                # Anchor refinement
                anchor_yaw_enable=True,
                anchor_yaw_strength=0.60,
                anchor_yaw_max_deg=70.0,
                anchor_min_clearance_m=0.30,
                anchor_yaw_min_fwd_clear_m=0.35,
                anchor_yaw_scan_deg=60.0,
                anchor_yaw_scan_steps=11,
                # Deterministic coverage
                deterministic_coverage=True,
                cov_offsets_deg=[20.0, -20.0, 160.0, -160.0],
                cov_max_retries=4,
                coverage_rounds=(0,),
                # Attempts / rejects
                max_attempts=32,
                max_rejects_per_visit=18,
            ),
            quality=dict(
                min_valid_frac=0.50,
                min_rel_depth_std=0.10,
                min_normal_disp=0.20,
                min_grad_mean=5.0,
                min_kpts=110,
                grid_nx=8, grid_ny=8,
                min_kpts_per_cell=0,
            ),
        ),

    Profile.MULTI_ROOMS_DENSE_2ROUND: dict(
        render=dict(
            H=1024, W=1024, SSAA=4, hfov_deg=95.0,
            pitch_start=-5, height_start_cm=150,
            round_pitch_deg=[-5.0, -8.0],   # slightly steeper 2nd pass
        ),
        coverage=dict(
            # Same intent as 1-round but with a complementary second pass
            n_views=350,
            traj_mode="video",
            n_anchors=24,
            traj_stride_m=0.45,
            min_anchor_sep_m=1.60,
            two_rounds=True,
            height2_cm=120.0,
            round_yaw_offset_deg=[0.0, 25.0],
            # Heading policy
            yaw_mode="mixed",
            yaw_mixed_alpha=0.35,
            yaw_limit_deg=24.0,
            yaw_post_limit_deg=12.0,
            yaw_smooth_alpha=0.35,
            # Anchor refinement
            anchor_yaw_enable=True,
            anchor_yaw_strength=0.60,
            anchor_yaw_max_deg=70.0,
            anchor_min_clearance_m=0.30,
            anchor_yaw_min_fwd_clear_m=0.35,
            anchor_yaw_scan_deg=60.0,
            anchor_yaw_scan_steps=11,
            # Deterministic coverage on both rounds; random follows after
            deterministic_coverage=True,
            cov_offsets_deg=[20.0, -20.0, 160.0, -160.0],
            cov_max_retries=4,
            coverage_rounds=(0, 1),
            # Attempts / rejects
            max_attempts=36,
            max_rejects_per_visit=20,
        ),
        quality=dict(
            min_valid_frac=0.50,
            min_rel_depth_std=0.10,
            min_normal_disp=0.20,
            min_grad_mean=5.0,
            min_kpts=110,
            grid_nx=8, grid_ny=8,
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