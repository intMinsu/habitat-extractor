import math
import numpy as np
import cv2

import habitat_sim
import magnum as mn

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi

def blend_yaw(y0: float, y1: float, alpha: float) -> float:
    # shortest-arc blend on circle
    d = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
    return wrap_pi(y0 + alpha * d)

def yaw_towards(src_xyz, dst_xyz):
    """
    Compute the yaw angle θ (in radians) about the world +Y axis that points from
    `src_xyz` to `dst_xyz` under Habitat’s right-handed world frame
    (X: right, Y: up, Z: forward).

    Convention:
        - 0 rad faces +Z (forward), positive yaw rotates toward +X.
        - Compatible with Habitat’s agent yaw via:
              quat_from_angle_axis(yaw, [0, 1, 0])

    Parameters
    ----------
    src_xyz : array-like, shape (3,)
        Source position [x, y, z] in world coordinates.
    dst_xyz : array-like, shape (3,)
        Target position [x, y, z] in world coordinates.

    Returns
    -------
    float
        Yaw angle θ in (-π, π], ignoring vertical offset (Δy).
    """
    dx = float(dst_xyz[0] - src_xyz[0])
    dz = float(dst_xyz[2] - src_xyz[2])
    return math.atan2(dx, dz)

def compute_path_yaws(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return np.zeros((points.shape[0],), dtype=np.float32)
    v = points[1:] - points[:-1]
    yaws = np.arctan2(v[:,0], v[:,2])  # dx, dz
    return np.concatenate([yaws[:1], yaws], axis=0).astype(np.float32)

def sample_target(sim, p_xyz, min_geo=2.0, max_geo=6.0, tries=50, seed=1234):
    """
    Sample a random navigable point on the navmesh whose **geodesic** distance from
    `p_xyz` lies within [min_geo, max_geo]. Returns the first point that satisfies
    this band, or None if not found within `tries`.

    Notes
    -----
    - Distance is the navmesh shortest-path length (geodesic), not Euclidean.
    - Randomness is controlled by `sim.pathfinder.seed(...)`. The `seed` argument
      here is informational; call `sim.pathfinder.seed(seed)` outside if needed.

    Parameters
    ----------
    sim : habitat_sim.Simulator
        Simulator with a loaded scene + navmesh.
    p_xyz : array-like, shape (3,)
        Start position [x, y, z] in world coordinates.
    min_geo : float, optional
        Minimum geodesic distance in meters (inclusive).
    max_geo : float, optional
        Maximum geodesic distance in meters (inclusive).
    tries : int, optional
        Maximum number of random samples to attempt.
    seed : int, optional
        (Informational) RNG seed; not used internally.

    Returns
    -------
    numpy.ndarray or None
        A 3-vector [x, y, z] for the sampled target point, or None if none found.
    """
    sp = habitat_sim.ShortestPath()
    sp.requested_start = mn.Vector3(float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2]))

    for _ in range(tries):
        q = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32).reshape(3)
        sp.requested_end = mn.Vector3(float(q[0]), float(q[1]), float(q[2]))
        if sim.pathfinder.find_path(sp) and (min_geo <= sp.geodesic_distance <= max_geo):
            return q
    return None


def normal_dispersion_metric(n_cam: np.ndarray, mask: np.ndarray, min_valid_px: int = 1000) -> float:
    """
    Compute spherical dispersion of normals D = 1 - ||mean(unit_normals)|| in [0,1].
    Lower means 'flatter'. Returns 0 if insufficient valid normals.
    """
    valid = mask & np.isfinite(n_cam).all(axis=2)
    v = n_cam[valid]
    if v.shape[0] < min_valid_px:
        return 0.0
    u = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    mvec = np.mean(u, axis=0)
    return float(1.0 - np.linalg.norm(mvec))


def is_good_view(depth, mask, n_cam, rgb, K,
                 min_valid_frac=0.65,
                 min_rel_depth_std=0.10,     # std(depth)/mean(depth)
                 min_normal_disp=0.12,       # dataset-dependent (Replica↑, HM3D↓)
                 min_grad_mean=6.0,          # mean |∇I| on valid mask (0..255)
                 min_kpts=600,               # ORB/SIFT total count
                 grid_nx=4, grid_ny=4, min_kpts_per_cell=5):
    """
    Decide if a rendered view is 'good' for downstream feature matching and triangulation
    using a combination of **geometry**, **photometry**, and **spatial coverage** tests.

    Geometry (depth & normals):
        • valid_frac ≥ min_valid_frac
        • mean(depth) not too near the camera
        • relative depth variation std/mean ≥ min_rel_depth_std (scale-free)
        • normal dispersion ≥ min_normal_disp  (diverse surface orientations)

    Photometry (texture proxies):
        • mean Sobel gradient on valid pixels ≥ min_grad_mean
        • ORB/SIFT keypoint count ≥ min_kpts

    Spatial coverage:
        • Divide the image into grid_nx × grid_ny cells. Require per-cell keypoint
          count ≥ min_kpts_per_cell in at least ~2/3 of the cells (configurable).

    Returns:
        True  if all gates pass
        False otherwise, printing the first failing reason for debugging.
    """
    valid = mask & np.isfinite(depth)
    valid_frac = float(valid.mean())
    print(f"[good_view] valid_frac={valid_frac:.3f} (min {min_valid_frac})")
    if valid_frac < min_valid_frac:
        print("[good_view] REJECT: not enough valid depth coverage.")
        return False

    vals = depth[valid]
    if vals.size < 500:
        print("[good_view] REJECT: too few valid depth samples.")
        return False

    # Scale-free depth variation
    z_mean = float(np.nanmean(vals))
    z_std  = float(np.nanstd(vals))
    rel_std = z_std / max(z_mean, 1e-6)
    print(f"[good_view] depth mean/std={z_mean:.3f}/{z_std:.3f}  rel_std={rel_std:.3f}  (min rel {min_rel_depth_std})")
    if z_mean < 0.9:
        print("[good_view] REJECT: scene too close (mean depth < 0.9 m).")
        return False
    if rel_std < min_rel_depth_std:
        print("[good_view] REJECT: too planar by relative depth std.")
        return False

    # Geometric diversity via normals
    disp = normal_dispersion_metric(n_cam, valid)
    print(f"[good_view] normal_dispersion={disp:.3f}  (min {min_normal_disp})")
    if disp < min_normal_disp:
        print("[good_view] REJECT: normals too flat.")
        return False

    # Photometric texture
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.mean((np.abs(gx) + np.abs(gy))[valid]))
    print(f"[good_view] grad_mean={grad_mean:.2f}  (min {min_grad_mean})")
    if grad_mean < min_grad_mean:
        print("[good_view] REJECT: image gradients too weak.")
        return False

    # Keypoints + coverage
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    kps = orb.detect(gray, None)
    k = len(kps)
    print(f"[good_view] keypoints={k}  (min {min_kpts})")
    if k < min_kpts:
        print("[good_view] REJECT: too few keypoints.")
        return False

    H, W = gray.shape
    counts = np.zeros((grid_ny, grid_nx), int)
    for kp in kps:
        x, y = kp.pt
        j = min(grid_nx - 1, max(0, int(x * grid_nx / W)))
        i = min(grid_ny - 1, max(0, int(y * grid_ny / H)))
        counts[i, j] += 1

    cells_failing = int((counts < min_kpts_per_cell).sum())
    total_cells   = grid_nx * grid_ny
    coverage_ok   = cells_failing <= total_cells // 3
    print(f"[good_view] per-cell≥{min_kpts_per_cell}: {(counts >= min_kpts_per_cell).sum()}/{total_cells}  "
          f"(allow fails ≤ {total_cells // 3})  -> coverage_ok={coverage_ok}")

    if not coverage_ok:
        print("[good_view] REJECT: poor spatial coverage of features.")
        return False

    print("[good_view] ACCEPT")
    return True

## Clearance helpers
def forward_clearance_m(pf, p_xyz, yaw, step=0.15, nsteps=40) -> float:
    """
    Returns how far (meters) one can walk straight from p_xyz along yaw before
    leaving the navmesh. Used as a proxy for 'open space ahead'.
    """
    x, y, z = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    sx, sz = math.sin(yaw), math.cos(yaw)
    for i in range(1, nsteps + 1):
        t = i * step
        q = mn.Vector3(x + sx * t, y, z + sz * t)
        if not pf.is_navigable(q):
            return (i - 1) * step
    return nsteps * step

def radial_clearance_m(sim, p_xyz) -> float:
    """
    Returns the navmesh distance (meters) from p_xyz to the closest obstacle,
    i.e. a scalar 'tightness' at the point.
    """
    try:
        return float(sim.pathfinder.distance_to_closest_obstacle(mn.Vector3(*map(float, p_xyz))))
    except Exception:
        return float("nan")

def refine_yaw_by_forward_clearance(pf, p_xyz, yaw0, *, min_clear=0.35, scan_deg=60.0, steps=11) -> float:
    """
    If forward clearance at yaw0 is below min_clear, scan ±scan_deg and pick the yaw
    with the best clearance (slightly penalizing larger turns).
    """
    c0 = forward_clearance_m(pf, p_xyz, yaw0)
    if c0 >= min_clear:
        return yaw0
    best_y, best_s = yaw0, -1e9
    scan_rad = math.radians(scan_deg)
    for t in np.linspace(-scan_rad, scan_rad, steps):
        y = wrap_pi(yaw0 + float(t))
        c = forward_clearance_m(pf, p_xyz, y)
        s = c - 0.05 * abs(t)  # prefer open space, but penalize big rotations
        if s > best_s:
            best_s, best_y = s, y
    return best_y


from habitat_sim.utils.common import quat_from_angle_axis
from .quaternion_helper import rotate_vec3_from_quat_axis

def set_agent_pose(agent, p_xyz, yaw_rad, pitch_rad, base_cam_height_m, cam_height_m):
    """
        Set the agent pose at world position `p_xyz` with yaw/pitch, adjusting for camera height.

        Yaw is applied about +Y (heading), then pitch about the current camera-right axis.
        Roll is not applied (keeps the camera level). The y-offset accounts for the
        difference between the height at which sensors were created (`base_cam_height_m`)
        and the desired per-frame camera height (`cam_height_m`).

        Args:
            agent: habitat_sim.Agent instance to update.
            p_xyz: (3,) array-like world position [x, y, z] in meters.
            yaw_rad: Heading in radians (0 faces −Z in Habitat’s convention).
            pitch_rad: Pitch in radians (negative looks slightly downward).
            base_cam_height_m: Sensor creation height in meters.
            cam_height_m: Desired camera height for this frame in meters.

        Side effects:
            Updates the agent state in-place via `agent.set_state(...)`.

        Example:
            # Sensors were created at 1.50 m, but we want this shot at 1.35 m,
            # with a slight downward pitch.
            import math
            p = (x, y, z)  # some navigable world position
            set_agent_pose(
                agent,
                p_xyz=p,
                yaw_rad=math.radians(30.0),
                pitch_rad=math.radians(-5.0),
                base_cam_height_m=1.50,
                cam_height_m=1.35,
            )
            obs = sim.get_sensor_observations()  # render with the new pose
    """
    base_y_offset = float(cam_height_m - base_cam_height_m)
    st = agent.get_state()
    st.position = mn.Vector3(float(p_xyz[0]), float(p_xyz[1] + base_y_offset), float(p_xyz[2]))
    q_yaw = quat_from_angle_axis(float(yaw_rad), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right = rotate_vec3_from_quat_axis((1.0, 0.0, 0.0), q_yaw)
    q_pitch = quat_from_angle_axis(float(pitch_rad), right)
    st.rotation = q_pitch * q_yaw
    agent.set_state(st)

def _score_yaw_openness_depth(sim, agent, p_xyz, yaw, *, cam_height_m, base_cam_height_m, pitch_rad, center_frac):
    """
       Depth-based 'forward openness' score at a pose/yaw.

       Places the agent at world position `p_xyz` with heading `yaw` and a small pitch,
       renders a depth image, then scores the *central vertical stripe* that spans
       `center_frac` of the image width. The score is:
           median_depth_in_stripe * (0.5 + 0.5 * valid_fraction_in_stripe).
       Higher is better (farther & more valid pixels straight ahead).

       Notes:
         • Temporarily modifies the agent pose and restores it before returning.
         • Use the same value for `base_cam_height_m` and `cam_height_m` unless you
           created sensors at a different fixed height and want a per-frame offset.

       Args:
           sim: habitat_sim.Simulator
           agent: habitat_sim.Agent
           p_xyz: (3,) world position (meters)
           yaw: heading in radians (0 faces −Z in Habitat)
           cam_height_m: desired per-frame camera height in meters
           base_cam_height_m: height used at sensor creation (meters)
           pitch_rad: small pitch (usually negative to look slightly downward)
           center_frac: horizontal fraction (0..1) for the scoring stripe

       Returns:
           float: openness score (higher means more forward free-space)

       Example:
           import math
           score = _score_yaw_openness_depth(
               sim, agent, p_xyz=(1.2, 0.0, 3.4), yaw=math.radians(30),
               cam_height_m=1.50, base_cam_height_m=1.50,
               pitch_rad=math.radians(-5), center_frac=0.28
           )
    """

    st_save = agent.get_state()
    set_agent_pose(agent, p_xyz, yaw, pitch_rad, base_cam_height_m, cam_height_m)
    obs = sim.get_sensor_observations()
    depth = obs["depth"].astype(np.float32)
    mask  = np.isfinite(depth) & (depth > 0)
    Hs, Ws = depth.shape
    w = max(4, int(round(center_frac * Ws)))
    j0 = max(0, Ws//2 - w//2); j1 = min(Ws, j0 + w)
    stripe = depth[:, j0:j1]; smask = mask[:, j0:j1]
    vals = np.where(smask, stripe, np.nan)
    med  = float(np.nanmedian(vals)) if np.any(smask) else 0.0
    vfrac = float(np.nanmean(smask)) if smask.size else 0.0
    agent.set_state(st_save)
    return med * (0.5 + 0.5 * vfrac)


def best_yaw_by_depthscan(sim, agent, p_xyz, yaw_seed, *, cam_height_m, base_cam_height_m, pitch_rad, scan_deg, steps, center_frac):
    """
        Pick the yaw around `yaw_seed` that maximizes forward openness.

        Scans `steps` headings uniformly in the range [yaw_seed ± scan_deg], scores
        each via `_score_yaw_openness_depth`, and returns the best yaw (radians).

        Args:
            sim, agent, p_xyz: see `_score_yaw_openness_depth`
            yaw_seed: center yaw (radians)
            cam_height_m, base_cam_height_m, pitch_rad, center_frac: forwarded args
            scan_deg: total ± range to scan (degrees)
            steps: number of yaw samples (>=3 recommended)

        Returns:
            float: best yaw (radians)

        Example:
            import math
            y_best = _best_yaw_by_depthscan(
                sim, agent, p_xyz=(1.2, 0.0, 3.4), yaw_seed=math.radians(15),
                cam_height_m=1.50, base_cam_height_m=1.50,
                pitch_rad=math.radians(-5), scan_deg=75.0, steps=9, center_frac=0.28
            )
    """
    steps = int(max(3, steps))
    scan = np.linspace(-math.radians(scan_deg), math.radians(scan_deg), steps)
    best_y, best_s = float(yaw_seed), -1e9
    for dt in scan:
        y = wrap_pi(yaw_seed + float(dt))
        s = _score_yaw_openness_depth(
            sim, agent, p_xyz, y,
            cam_height_m=cam_height_m,
            base_cam_height_m=base_cam_height_m,
            pitch_rad=pitch_rad,
            center_frac=center_frac,
        )
        if s > best_s:
            best_s, best_y = s, y
    return best_y




