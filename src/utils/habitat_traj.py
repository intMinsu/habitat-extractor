# utils/habitat_traj.py
import math
from typing import List, Tuple
import numpy as np
import habitat_sim
import magnum as mn

from src.utils.habitat_sample import sample_target

def _geo_dist(pf: habitat_sim.PathFinder, a: np.ndarray, b: np.ndarray) -> float:
    sp = habitat_sim.ShortestPath()
    sp.requested_start = mn.Vector3(float(a[0]), float(a[1]), float(a[2]))
    sp.requested_end   = mn.Vector3(float(b[0]), float(b[1]), float(b[2]))
    if pf.find_path(sp):
        return float(sp.geodesic_distance)
    return float("inf")

def _geo_path_points(pf: habitat_sim.PathFinder, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a polyline (N,3) along the navmesh shortest path from a to b (incl. endpoints)."""
    sp = habitat_sim.ShortestPath()
    sp.requested_start = mn.Vector3(float(a[0]), float(a[1]), float(a[2]))
    sp.requested_end   = mn.Vector3(float(b[0]), float(b[1]), float(b[2]))
    ok = pf.find_path(sp)
    if not ok or sp.geodesic_distance == float("inf"):
        return np.stack([a, b], 0)
    pts = np.array([[p[0], p[1], p[2]] for p in sp.points], dtype=np.float32) if len(sp.points) > 0 else np.stack([a, b], 0)
    return pts

def _far_score(sim, p, min_geo, max_geo, trials=12):
    """Proxy for 'can this spot see far?': try a few target picks and keep the best geodesic d."""
    pf = sim.pathfinder
    best = 0.0
    for _ in range(trials):
        q = sample_target(sim, p, min_geo=min_geo, max_geo=max_geo, tries=32)
        if q is not None:
            d = _geo_dist(pf, p, q)  # geodesic distance to that target
            if np.isfinite(d):
                best = max(best, float(d))
    return best

def _resample_polyline(points: np.ndarray, stride_m: float) -> np.ndarray:
    """Resample a 3D polyline at (approximately) fixed arc-length spacing."""
    if points.shape[0] <= 1:
        return points.copy()
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arclen[-1])
    if total == 0.0:
        return points[:1].copy()

    t_samples = np.arange(0.0, total, stride_m, dtype=np.float32)
    out = []
    j = 0
    for t in t_samples:
        while j+1 < len(arclen) and arclen[j+1] < t:
            j += 1
        if j+1 >= len(arclen):  # clamp
            out.append(points[-1])
            break
        denom = max(arclen[j+1] - arclen[j], 1e-9)
        alpha = (t - arclen[j]) / denom
        out.append((1.0 - alpha) * points[j] + alpha * points[j+1])
    if np.linalg.norm(out[-1] - points[-1]) > 1e-4:
        out.append(points[-1])
    return np.asarray(out, dtype=np.float32)

def sample_anchors_fps(
    sim: habitat_sim.Simulator,
    n_anchors: int,
    *,
    tries_per_anchor: int = 200,
    min_sep_geo: float = 2.0,
    seed: int = 1234,
    max_dy: float = 0.5,
    min_clearance: float = 0.0,
    vantage_lambda: float = 0.0,
    vantage_min_geo: float = 3.0,
    vantage_max_geo: float = 10.0,
    vantage_trials: int = 12,
) -> List[np.ndarray]:
    """
        Geodesic farthest-point sampling of anchors on the navmesh (with clearance filter).

        Algorithm:
          • Randomly propose navigable points; accept those that:
              - have radial clearance ≥ `min_clearance`,
              - lie within vertical span `±max_dy` of the first anchor,
              - are geodesically ≥ `min_sep_geo` from all accepted anchors.
          • Score candidates by min geodesic distance to existing anchors.
            Optionally add a “vantage” bonus proportional to forward emptiness
            (probing random targets in [vantage_min_geo, vantage_max_geo]).
          • If no candidate meets thresholds, gradually relax separation/clearance.

        Args:
            sim: Habitat simulator.
            n_anchors: desired number of anchors.
            tries_per_anchor: proposals per new anchor before relaxing thresholds.
            min_sep_geo: minimum geodesic separation (meters) between anchors.
            seed: RNG seed.
            max_dy: allowed vertical deviation relative to the first accepted anchor.
            min_clearance: min radial clearance (m) to nearest obstacle (0 disables).
            vantage_lambda: weight for the vantage bonus (0 disables).
            vantage_min_geo / vantage_max_geo / vantage_trials: vantage probe settings.

        Returns:
            List of (3,) float32 world positions.

        Example:
            anchors = sample_anchors_fps(
                sim, n_anchors=18, seed=0,
                tries_per_anchor=300, min_sep_geo=2.0,
                min_clearance=0.35,     # ← matches anchor_min_clearance_m
                vantage_lambda=0.30,    # ← slightly prefers “open-view” anchors
            )
    """
    rng = np.random.default_rng(seed)
    pf = sim.pathfinder

    def clearance_ok(q):
        try:
            c = pf.distance_to_closest_obstacle(mn.Vector3(float(q[0]), float(q[1]), float(q[2])))
            return float(c) >= float(min_clearance_cur)
        except Exception:
            return True

    def rand_nav():
        return np.asarray(pf.get_random_navigable_point(), dtype=np.float32).reshape(3)

    anchors: List[np.ndarray] = []

    # First seed (respecting clearance)
    min_clearance_cur = float(min_clearance)
    for _ in range(max(tries_per_anchor, 200)):
        p0 = rand_nav()
        if np.isfinite(p0).all() and clearance_ok(p0):
            anchors.append(p0)
            break
    if not anchors:
        return anchors

    # Auto-relax loop parameters
    sep_cur = float(min_sep_geo)
    relax_round = 0

    while len(anchors) < n_anchors:
        best_q, best_score = None, -1.0
        y0 = anchors[0][1]
        found_this_round = False

        for _ in range(int(tries_per_anchor * (1.0 + 0.25 * relax_round))):
            q = rand_nav()
            if (not np.isfinite(q).all()
                or abs(float(q[1] - y0)) > max_dy
                or not clearance_ok(q)):
                continue

            # geodesic min-separation to existing anchors
            ok = True
            dmins = []
            for a in anchors:
                d = _geo_dist(pf, a, q)
                if not np.isfinite(d) or d < sep_cur:
                    ok = False
                    break
                dmins.append(d)
            if not ok:
                continue

            coverage = float(np.min(dmins))

            vbonus = 0.0
            if vantage_lambda > 0.0:
                v = _far_score(sim, q, min_geo=vantage_min_geo, max_geo=vantage_max_geo, trials=vantage_trials)
                vbonus = float(vantage_lambda) * float(v)

            score = coverage + vbonus
            if score > best_score:
                best_score, best_q = score, q
                found_this_round = True

        if best_q is not None:
            anchors.append(best_q)
            # after a success, gently restore toward the original thresholds
            sep_cur = min(sep_cur * 1.05, float(min_sep_geo))
            min_clearance_cur = min(min_clearance_cur * 1.05, float(min_clearance))
            relax_round = 0
            continue

        # --- No candidate found: relax and try again ---
        relax_round += 1
        # soften both knobs a bit (capped)
        sep_cur *= 0.9
        min_clearance_cur *= 0.9
        # stop if we’re really tiny
        if sep_cur < 0.5 * max(0.6, min_sep_geo) and min_clearance_cur < 0.6 * max(0.10, min_clearance):
            break

    return anchors


def order_anchors_greedy(sim: habitat_sim.Simulator, anchors: List[np.ndarray], closed: bool = True) -> List[int]:
    """
    Greedy TSP tour over anchors under geodesic distance:
    start at 0, repeatedly append the nearest unvisited. If `closed`, the
    last will connect back to the first (we’ll add that segment later).
    """
    pf = sim.pathfinder
    N = len(anchors)
    if N <= 2:
        return list(range(N))

    remain = set(range(1, N))
    order = [0]
    cur = 0
    while remain:
        best_j, best_d = None, float("inf")
        for j in list(remain):
            d = _geo_dist(pf, anchors[cur], anchors[j])
            if d < best_d:
                best_d, best_j = d, j
        order.append(best_j)
        remain.remove(best_j)
        cur = best_j
    return order

def _densify_by_curvature(points: np.ndarray, max_seg: float = 0.30) -> np.ndarray:
    """
    Insert intermediate samples so consecutive points are at most ~max_seg apart.
    This ensures smoothing preserves tight turns faithfully.
    """
    if points.shape[0] <= 1:
        return points
    out = [points[0]]
    for i in range(1, points.shape[0]):
        a, b = points[i-1], points[i]
        d = float(np.linalg.norm(b - a))
        n_mid = max(0, int(math.floor(d / max_seg)))
        if n_mid > 0:
            alphas = (np.arange(1, n_mid+1, dtype=np.float32) / (n_mid+1)).reshape(-1, 1)
            mids = (1.0 - alphas) * a + alphas * b
            out.append(mids)
        out.append(b)
    return np.concatenate([p if p.ndim == 2 else p[None] for p in out], axis=0).astype(np.float32)

def _chaikin_smooth(points: np.ndarray, passes: int = 2, keep_ends: bool = True) -> np.ndarray:
    """
    Chaikin corner cutting. Each pass reduces corners, approximating a C1 curve.
    """
    if points.shape[0] < 3 or passes <= 0:
        return points.copy()
    P = points.copy()
    for _ in range(passes):
        Q = []
        n = P.shape[0]
        if keep_ends:
            Q.append(P[0])
        for i in range(n-1):
            p, r = P[i], P[i+1]
            q = 0.75 * p + 0.25 * r
            s = 0.25 * p + 0.75 * r
            Q.extend([q, s])
        if keep_ends:
            Q.append(P[-1])
        P = np.asarray(Q, dtype=np.float32)
    return P

def _clearance_score(pf: habitat_sim.PathFinder, p: np.ndarray, *, radius: float = 0.6, dirs: int = 16) -> float:
    """
    Fraction of directions on a ring of radius `radius` that are geodesically reachable
    at ≥ 0.8*radius (proxy for 'open space' around p).
    """
    ang = np.linspace(0, 2*math.pi, dirs, endpoint=False)
    good = 0
    for a in ang:
        target = p + np.array([math.sin(a)*radius, 0.0, math.cos(a)*radius], dtype=np.float32)
        d = _geo_dist(pf, p, target)
        if np.isfinite(d) and d >= 0.8*radius:
            good += 1
    return good / max(dirs, 1)

def limit_yaw_rate(yaws: np.ndarray, max_rate_deg: float = 12.0) -> np.ndarray:
    """
    Smooth heading changes by clamping per-step Δyaw.

    Unwraps `yaws` (radians), limits the difference between consecutive samples
    to ±`max_rate_deg`, then re-wraps to (-π, π].

    Args:
        yaws: (N,) array of yaw angles in radians.
        max_rate_deg: max allowed change (degrees) between successive samples.

    Returns:
        (N,) array of rate-limited yaws (radians).

    Example:
        import numpy as np, math
        y = np.array([0.0, math.radians(90), math.radians(180)], np.float32)
        y_smooth = limit_yaw_rate(y, max_rate_deg=20)  # caps each step to ±20°
    """
    if yaws.size <= 1:
        return yaws
    # unwrap
    y = np.unwrap(yaws)
    max_rate = math.radians(max_rate_deg)
    for i in range(1, y.size):
        dy = y[i] - y[i-1]
        if dy >  max_rate: y[i] = y[i-1] + max_rate
        if dy < -max_rate: y[i] = y[i-1] - max_rate
    # rewrap to (-π, π]
    return ( (y + math.pi) % (2*math.pi) ) - math.pi

def make_video_like_trajectory(
    sim: habitat_sim.Simulator,
    n_anchors: int,
    stride_m: float,
    *,
    seed: int = 1234,
    tries_per_anchor: int = 200,
    min_sep_geo: float = 2.0,
    closed_loop: bool = True,
    # --- smoothing knobs ---
    densify_max_seg_m: float = 0.30,
    smooth_passes: int = 2,
    return_debug: bool = False,
    anchor_min_clearance_m: float = 0.35
):
    # -> np.ndarray | tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Build a smooth, “video-like” base trajectory on the navmesh.

    Steps:
      1) Sample `n_anchors` informative, well-separated anchors on the navmesh
         (geodesic FPS + optional vantage bonus), requiring local radial clearance
         ≥ `anchor_min_clearance_m`.
      2) Order anchors into a geodesic tour (optionally closed).
      3) Connect anchors with pathfinder polylines, resample at `stride_m`.
      4) Densify tight corners and Chaikin-smooth, then enforce final `stride_m`.

    Args:
        sim: Habitat simulator.
        n_anchors: how many anchors to place.
        stride_m: target arc-length spacing (meters) for the final path.
        seed, tries_per_anchor, min_sep_geo, closed_loop: anchor sampling/tour knobs.
        densify_max_seg_m, smooth_passes: curvature densification + smoothing.
        return_debug: if True, also return anchor list and raw polylines.
        anchor_min_clearance_m: minimum radial clearance (m) an anchor must have.

    Returns:
        traj: (M,3) float32 world positions.
        If return_debug=True: (traj, anchors: List[np.ndarray], raw_polylines: List[np.ndarray])

    Example:
        traj, anchors, polys = make_video_like_trajectory(
            sim, n_anchors=18, stride_m=0.40, seed=0,
            min_sep_geo=2.0, closed_loop=True,
            densify_max_seg_m=0.25, smooth_passes=1,
            return_debug=True, anchor_min_clearance_m=0.35
        )
    """

    # in make_video_like_trajectory(...)
    anchors = sample_anchors_fps(
        sim, n_anchors,
        tries_per_anchor=400,  # a touch higher is cheap insurance
        min_sep_geo=min_sep_geo,
        seed=seed,
        max_dy=1.0,  # was 0.5
        min_clearance=anchor_min_clearance_m,
        vantage_lambda=0.30,
        vantage_min_geo=3.0,
        vantage_max_geo=10.0,
        vantage_trials=12,
    )

    if len(anchors) == 0:
        return (np.zeros((0, 3), np.float32), anchors, []) if return_debug else np.zeros((0, 3), np.float32)

    order = order_anchors_greedy(sim, anchors, closed=closed_loop)
    pf = sim.pathfinder

    raw_polys: list[np.ndarray] = []
    traj_pts: list[np.ndarray] = []

    for i in range(len(order)):
        a = anchors[order[i]]
        b = anchors[order[(i+1) % len(order)]] if closed_loop or i+1 < len(order) else None
        traj_pts.append(a)
        if b is None:
            continue
        poly = _geo_path_points(pf, a, b)
        raw_polys.append(poly)
        seg_pts = _resample_polyline(poly, stride_m)  # coarse
        if seg_pts.shape[0] > 1:
            traj_pts.append(seg_pts[1:])

    traj = np.concatenate([p if p.ndim == 2 else p[None, ...] for p in traj_pts], axis=0).astype(np.float32)

    # --- densify tight corners then smooth ---
    max_seg = max(densify_max_seg_m, stride_m)
    traj = _densify_by_curvature(traj, max_seg=max_seg)
    traj = _chaikin_smooth(traj, passes=smooth_passes, keep_ends=True)

    traj = _resample_polyline(traj, stride_m)  # final enforce
    # print(f"traj = {traj}")

    return (traj, anchors, raw_polys) if return_debug else traj

def compute_anchor_headings(
    sim,
    agent,
    base_traj: np.ndarray,
    anchors_list,
    base_yaws: np.ndarray,
    *,
    pitch_deg: float,
    viz_scan_deg: float,
    viz_scan_steps: int,
    center_frac: float,
    min_fwd_clear_m: float,
    yaw_scan_deg: float,
    yaw_scan_steps: int,
    cam_height_start_m: float,
):
    """
    Refine path-tangent headings at anchors using two complementary signals:

      • Depth openness (render-based): look where the camera “sees” farther;
        what the camera “sees” along the optical axis.
      • Forward clearance (navmesh-based): keep the heading walkable;
        how far you can safely walk straight ahead on the navmesh.

    Per-anchor pipeline
    -------------------
      1) Map the anchor to its nearest path sample (dedupe & sort).
      2) Seed yaw = path tangent at that sample (base_yaws[idx]).
      3) Depth openness scan around the seed (±viz_scan_deg over viz_scan_steps).
      4) If forward clearance < min_fwd_clear_m, clamp by searching ±yaw_scan_deg
         (yaw_scan_steps) on the navmesh and picking a safer yaw.
      5) Also report local *radial* clearance (meters) at that point.

    Disabling pieces
    ----------------
      • Disable clearance clamp: set min_fwd_clear_m <= 0.
      • Disable depth scan: set viz_scan_deg = 0.0 (the scan collapses to seed yaw).

    Args:
        sim, agent: Habitat simulator & agent.
        base_traj: (M,3) path samples in world coordinates.
        anchors_list: list/array of (x,y,z) world anchors.
        base_yaws: (M,) path-tangent yaws (radians).
        pitch_deg: small pitch used during depth scoring (deg; e.g. −5).
        viz_scan_deg, viz_scan_steps, center_frac: depth-scan knobs.
        min_fwd_clear_m: forward-clear threshold (meters).
        yaw_scan_deg, yaw_scan_steps: clearance-clamp scan range/steps.
        cam_height_start_m: camera height used for scoring (meters).

    Returns:
        anchor_idxs : (Na,) int indices into base_traj.
        seed_yaws   : (Na,) float32 seed (tangent) yaws [rad].
        best_yaws   : (Na,) float32 refined yaws [rad].
        local_clr_m : (Na,) float32 radial clearance at anchors [m].

    Example:
        import math, numpy as np
        anchor_idxs, seed_yaws, best_yaws, local_clr_m = compute_anchor_headings(
            sim, agent,
            base_traj=path_xyz,                        # (M,3)
            anchors_list=anchor_pts,                   # list/array of XYZ
            base_yaws=compute_path_yaws(path_xyz),     # (M,)
            pitch_deg=-5.0,
            viz_scan_deg=75.0, viz_scan_steps=9, center_frac=0.28,
            min_fwd_clear_m=0.35, yaw_scan_deg=60.0, yaw_scan_steps=11,
            cam_height_start_m=1.50,
        )
        # Tip: to disable the depth scan, pass viz_scan_deg=0.0;
        # to disable the clamp, pass min_fwd_clear_m=0.0.
    """
    # Map anchors -> nearest path index (dedup, sorted)
    if isinstance(anchors_list, (list, tuple)) and len(anchors_list):
        anchors_np = np.asarray(anchors_list, np.float32)
        anchor_idxs, seen = [], set()
        for a in anchors_np:
            d2 = np.sum((base_traj - a[None, :]) ** 2, axis=1)
            idx = int(np.argmin(d2))
            if idx not in seen:
                seen.add(idx); anchor_idxs.append(idx)
        anchor_idxs = np.asarray(sorted(anchor_idxs), int)
    else:
        # Fallback: use every path sample as an "anchor" (rare)
        anchor_idxs = np.arange(base_traj.shape[0], dtype=int)

    seed_yaws = base_yaws[anchor_idxs]
    H_cam = float(cam_height_start_m)
    pitch_rad = math.radians(float(pitch_deg))

    best_yaws, local_clr = [], []

    from .habitat_sample import (
        best_yaw_by_depthscan,
        refine_yaw_by_forward_clearance,
        radial_clearance_m
    )

    for k, idx in enumerate(anchor_idxs):
        p = base_traj[idx]
        y_seed = float(seed_yaws[k])

        # Depth-based openness scan (skip if disabled)
        if viz_scan_steps <= 0 or viz_scan_deg <= 0:
            y_best = y_seed
        else:
            y_best = best_yaw_by_depthscan(
                sim, agent, p, y_seed,
                cam_height_m=H_cam,
                base_cam_height_m=H_cam,
                pitch_rad=pitch_rad,
                scan_deg=float(viz_scan_deg),
                steps=int(viz_scan_steps),
                center_frac=float(center_frac),
            )

        # Forward-clearance clamp (skip if disabled)
        if float(min_fwd_clear_m) > 0:
            y_best = refine_yaw_by_forward_clearance(
                sim.pathfinder, p_xyz=p, yaw0=y_best,
                min_clear=float(min_fwd_clear_m),
                scan_deg=float(yaw_scan_deg), steps=int(yaw_scan_steps),
            )

        best_yaws.append(y_best)
        local_clr.append(radial_clearance_m(sim, p))

    return (
        anchor_idxs.astype(int),
        seed_yaws.astype(np.float32),
        np.asarray(best_yaws, np.float32),
        np.asarray(local_clr,  np.float32),
    )



# --- Anchor & yaw helpers -----------------------------------------------------
def _interp_circular_1d(N: int, key_idx_vals):
    """
    Interpolate scalar values on a circular index domain [0..N-1].
    key_idx_vals: list of (idx, value) pairs. Values are already 'linear' (not wrapped).
    Returns array length N.
    """
    out = np.zeros((N,), np.float32)
    if len(key_idx_vals) == 0:
        return out
    key_idx_vals = sorted(key_idx_vals, key=lambda t: t[0])
    for k in range(len(key_idx_vals)):
        i, vi = key_idx_vals[k]
        j, vj = key_idx_vals[(k + 1) % len(key_idx_vals)]
        # walk forward around the ring
        j_wrap = j if j > i else j + N
        length = j_wrap - i
        if length <= 0:
            out[i] = vi
            continue
        for t in range(length + 1):
            u = float(t) / float(max(1, length))
            val = (1.0 - u) * vi + u * vj
            out[(i + t) % N] = val
    return out

def build_yaw_schedule_from_anchors(
    base_yaws: np.ndarray,
    anchor_idxs: np.ndarray,
    best_yaws: np.ndarray,
    *,
    strength: float,
    max_delta_deg: float,
    yaw_limit_deg: float,
) -> np.ndarray:
    """
    Build a full path yaw schedule from per-anchor refinements.

    Steps:
      1) At each anchor i: delta_i = wrap(best_yaws[i] − base_yaws[idx_i]),
         clamped to ±max_delta_deg.
      2) Circularly interpolate {idx_i → delta_i} over [0..N−1].
      3) Apply blend strength ∈ [0,1] to mix with the base.
      4) Optionally yaw-rate-limit for smoothness.

    Args:
        base_yaws: (N,) base/path-tangent yaws [rad].
        anchor_idxs: (Na,) unique, sorted indices into the path (recommended).
        best_yaws: (Na,) refined per-anchor yaws [rad].
        strength: blend factor in [0,1]; 0 = base only, 1 = fully apply deltas.
        max_delta_deg: clamp per-anchor delta magnitude (deg).
        yaw_limit_deg: max per-step change after blending (deg); ≤0 disables.

    Returns:
        (N,) np.ndarray yaw schedule [rad].

    Notes:
        anchor_yaw_max_deg
        Per-anchor clamp on how far the refined “best” yaw is allowed to deviate
        from the path-tangent yaw. Think of it as an outlier guard: even if the
        depth/clearance logic suggests a big turn, we limit the injected delta
        to ±anchor_yaw_max_deg before we interpolate over the whole path.

        yaw_limit_deg
        A rate limiter on the per-step yaw change along the path. After we create
        a yaw sequence (path tangent or refined schedule), we cap Δyaw between
        consecutive samples to keep orientation changes smooth and video-like.
        Higher values = snappier turns; lower values = gentler motion.

    Example:
        import numpy as np, math
        N = 200
        base = np.linspace(-0.5, 0.5, N).astype(np.float32)      # toy path tangent
        aidx = np.array([20, 80, 140], int)
        best = base[aidx] + np.deg2rad([30, -20, 15])
        sched = build_yaw_schedule_from_anchors(
            base_yaws=base, anchor_idxs=aidx, best_yaws=best,
            strength=0.6, max_delta_deg=45.0, yaw_limit_deg=25.0,
        )
    """
    key_idx_delta = []
    max_delta = math.radians(float(max_delta_deg))
    for k, idx in enumerate(anchor_idxs):
        d = math.atan2(
            math.sin(float(best_yaws[k]) - float(base_yaws[idx])),
            math.cos(float(best_yaws[k]) - float(base_yaws[idx])),
        )
        key_idx_delta.append((int(idx), float(np.clip(d, -max_delta, max_delta))))

    delta_full = _interp_circular_1d(base_yaws.shape[0], key_idx_delta)
    yaw_sched = base_yaws + float(np.clip(strength, 0.0, 1.0)) * delta_full
    yaw_sched = ((np.unwrap(yaw_sched) + math.pi) % (2 * math.pi)) - math.pi
    if float(yaw_limit_deg) > 0:
        yaw_sched = limit_yaw_rate(yaw_sched, max_rate_deg=float(yaw_limit_deg))
    return yaw_sched

def plan_coverage_playlist(
    anchor_idxs: np.ndarray,
    best_yaws: np.ndarray,
    cov_offsets_deg,
    *,
    # New knobs to avoid “spin-in-place” sequences
    order: str = "offset-major",        # {"offset-major","anchor-major(legacy)"}
    vi_stride: int = 1,                 # how far to shift anchors between offsets
    avoid_pure_rotation: bool = True,   # interleave anchors so rotations are not consecutive
    path: np.ndarray = None,            # (Nv,3) visit/path points; enables baseline enforcement
    min_trans_baseline_m: float = 0.0,  # e.g. 0.25 to ensure >=25 cm between consecutive plans
) -> list:
    """
    Deterministic coverage shots near anchors without back-to-back pure rotations.

    Strategy
    --------
    • offset-major order (default): iterate offsets first, then anchors. This naturally
      alternates positions between consecutive items (translation!), vs. the legacy
      anchor-major which emits multiple yaws at the SAME anchor back-to-back.
    • If `avoid_pure_rotation=True`, we roll the anchor list by `vi_stride` for each
      successive offset so each offset is taken at a neighboring anchor.
    • Optional baseline guard: if `path` is given, enforce at least
      `min_trans_baseline_m` between consecutive planned positions.

    Args
    ----
    anchor_idxs     : (Na,) indices into the path (e.g., anchor waypoints)
    best_yaws       : (Na,) refined yaw per anchor [rad]
    cov_offsets_deg : iterable of offsets (deg), e.g. [0, 20, -20]
    order           : "offset-major" (recommended) or "anchor-major" (legacy behavior)
    vi_stride       : how many anchors to advance for each successive offset (>=1)
    avoid_pure_rotation : interleave offsets across anchors to avoid spin-in-place
    path            : optional (Nv,3) world positions for baseline checks
    min_trans_baseline_m : minimum Euclidean distance between consecutive plan positions

    Returns
    -------
    list of dicts: [{"vi": int, "yaw": float_rad}, ...]
    """
    import math
    import numpy as np
    from .habitat_sample import wrap_pi

    anchor_idxs = np.asarray(anchor_idxs, dtype=int)
    best_yaws   = np.asarray(best_yaws, dtype=np.float32)
    offsets_deg = list(cov_offsets_deg or [0.0])

    if anchor_idxs.size == 0:
        return []

    playlist = []

    if order == "anchor-major":
        # Legacy: multiple yaws at the same anchor consecutively (rotation heavy)
        for k, idx in enumerate(anchor_idxs):
            base = float(best_yaws[k])
            for off in offsets_deg:
                playlist.append({"vi": int(idx), "yaw": wrap_pi(base + math.radians(float(off)))})
    else:
        # Recommended: offsets first, then anchors (translation between consecutive items)
        Na = anchor_idxs.size
        for j, off in enumerate(offsets_deg):
            # shift anchors by j*vi_stride so each offset is taken at a different anchor
            if avoid_pure_rotation and Na > 1:
                shift = int(j * max(1, vi_stride)) % Na
                a_idx = np.roll(anchor_idxs, -shift)
                a_yaw = np.roll(best_yaws,  -shift)
            else:
                a_idx = anchor_idxs
                a_yaw = best_yaws
            yaw_add = math.radians(float(off))
            for idx, base in zip(a_idx, a_yaw):
                playlist.append({"vi": int(idx), "yaw": wrap_pi(float(base) + yaw_add)})

    # Optional: enforce a minimum translational baseline between consecutive plans
    if path is not None and float(min_trans_baseline_m) > 0.0 and len(playlist) > 1:
        P = np.asarray(path, dtype=np.float32)
        Na = anchor_idxs.size
        # map vi->position in anchor sequence for quick neighbor lookup
        pos_in_anchor_ring = {int(a): i for i, a in enumerate(anchor_idxs.tolist())}
        fixed = [playlist[0]]
        for item in playlist[1:]:
            vi = item["vi"]
            ok = True
            if 0 <= vi < P.shape[0]:
                d = np.linalg.norm(P[vi] - P[fixed[-1]["vi"]])
                ok = (d >= float(min_trans_baseline_m))
            if not ok and Na > 1:
                # walk forward in the anchor ring until baseline is met (or give up)
                start = pos_in_anchor_ring.get(vi, 0)
                for step in range(1, Na):
                    vi2 = int(anchor_idxs[(start + step) % Na])
                    if 0 <= vi2 < P.shape[0]:
                        d2 = np.linalg.norm(P[vi2] - P[fixed[-1]["vi"]])
                        if d2 >= float(min_trans_baseline_m):
                            vi = vi2
                            break
                item = {"vi": vi, "yaw": item["yaw"]}
            fixed.append(item)
        playlist = fixed

    return playlist