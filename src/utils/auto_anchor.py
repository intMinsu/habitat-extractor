# src/utils/auto_anchor.py
import numpy as np
from dataclasses import replace

try:
    import habitat_sim  # required for ShortestPath
except ImportError as e:
    raise RuntimeError("auto_anchor requires habitat_sim (Habitat-Sim).") from e


def _geo_distance(pathfinder, a, b):
    """
    Geodesic distance along the navmesh between a and b.
    Falls back to np.inf if no path exists.
    """
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    ok = pathfinder.find_path(sp)
    if ok and np.isfinite(sp.geodesic_distance) and sp.geodesic_distance > 0:
        return float(sp.geodesic_distance)
    return np.inf


def estimate_navmesh_span(sim, n=512, pair_budget=1024):
    """
    Probe the navmesh to estimate a robust geodesic 'span' D.
    Returns (D, dy, (bb_min, bb_max)):
      D      : ~95th percentile geodesic distance between random nav points (meters)
      dy     : vertical span of sampled points
      bb_*   : axis-aligned bbox of the samples (xyz)
    """
    pf = sim.pathfinder
    pts = []
    for _ in range(n):
        p = np.asarray(pf.get_random_navigable_point(), dtype=np.float32).reshape(3)
        if np.isfinite(p).all():
            pts.append(p)
    if len(pts) < 2:
        return 0.0, 0.0, (np.zeros(3, np.float32), np.zeros(3, np.float32))

    P = np.stack(pts, 0)
    bb_min, bb_max = P.min(0), P.max(0)
    dy = float(bb_max[1] - bb_min[1])

    # Sample random pairs and compute geodesic distances
    rng = np.random.default_rng(0)
    m = min(pair_budget, len(P) * 4)
    pairs = rng.integers(0, len(P), size=(m, 2))
    dists = []
    for i, j in pairs:
        if i == j:
            continue
        dij = _geo_distance(pf, P[i], P[j])
        if np.isfinite(dij):
            dists.append(dij)

    if dists:
        D = float(np.percentile(dists, 95))
    else:
        # No valid geodesics found (very unlikely) → fallback to Euclidean diag in XZ
        diag = np.linalg.norm((bb_max[[0, 2]] - bb_min[[0, 2]]), ord=2)
        D = float(diag)

    return D, dy, (bb_min.astype(np.float32), bb_max.astype(np.float32))


def estimate_clearance_stats(sim, n=512):
    """
    Probe isotropic obstacle clearance at random nav points.
    Returns (p10, median, p90). If API missing, returns zeros.
    """
    pf = sim.pathfinder
    vals = []
    for _ in range(n):
        q = np.asarray(pf.get_random_navigable_point(), dtype=np.float32).reshape(3)
        try:
            # Habitat-Sim API name (present in modern builds)
            c = pf.distance_to_closest_obstacle(q)
        except Exception:
            # Older builds may not expose this; give up gracefully
            c = np.nan
        if np.isfinite(c):
            vals.append(float(c))

    if not vals:
        return 0.0, 0.0, 0.0
    v = np.array(vals, np.float32)
    return float(np.percentile(v, 10)), float(np.median(v)), float(np.percentile(v, 90))


from dataclasses import replace
import numpy as np

def auto_tune_anchor_params(sim, cov):
    """
    Pick sensible anchor spacing/clearance for *this* scene.

    Heuristic:
      • Estimate navmesh span (approx. XY range) and vertical range.
      • Set min_anchor_sep_m ≈ 0.8 * span/(n_anchors-1) (clamped to [1.0, 3.0] m).
      • Relax anchor_min_clearance_m toward ~max(0.20 m, 0.5 * median obstacle distance).
      • If still span-limited, cap n_anchors to what fits at the chosen separation.

    Returns
      cov2  : CoverageCfg with tuned fields (dataclasses.replace(cov, ...))
      notes : list[str] human-readable messages for logging
    """
    D, dy, (bb_min, bb_max) = estimate_navmesh_span(sim)
    q10, q50, q90 = estimate_clearance_stats(sim)

    notes = []
    cov2 = cov

    if D <= 0.0:
        notes.append("navmesh span unavailable; keeping defaults")
        return cov2, notes

    # scene scale summary (user-friendly: spell out what D means)
    notes.append(f"scene scale → navmesh span ≈ {D:.1f} m (XY), vertical range ≈ {dy:.1f} m (Y)")

    # Ideal 1D separation along a loop of length ~D is ~D/(n-1); keep 0.8 safety factor
    if cov.n_anchors > 1:
        sep_target = 0.8 * (D / (cov.n_anchors - 1))
    else:
        sep_target = cov.min_anchor_sep_m

    # Reasonable indoor envelope
    sep_lo, sep_hi = 1.0, 3.0
    sep_new = float(np.clip(sep_target, sep_lo, sep_hi))

    if abs(sep_new - cov.min_anchor_sep_m) / max(1e-6, cov.min_anchor_sep_m) > 0.1:
        notes.append(
            f"min_anchor_sep_m {cov.min_anchor_sep_m:.2f} → {sep_new:.2f} m "
            f"(based on navmesh span ≈ {D:.1f} m)"
        )
        cov2 = replace(cov2, min_anchor_sep_m=sep_new)

    # Clearance: relax toward ~max(0.20, 0.5*median_clearance) in tight scenes
    if q50 > 0:
        clr_target = max(0.20, 0.5 * q50)
        if clr_target < cov.anchor_min_clearance_m:
            notes.append(
                f"anchor_min_clearance_m {cov.anchor_min_clearance_m:.2f} → {clr_target:.2f} m "
                f"(median obstacle distance ≈ {q50:.2f} m)"
            )
            cov2 = replace(cov2, anchor_min_clearance_m=clr_target)

    # Upper bound on distinct anchors we can place given span and separation
    n_max_line = int(max(2, np.floor(D / max(0.5, cov2.min_anchor_sep_m)) + 1))
    if cov2.n_anchors > n_max_line:
        notes.append(
            f"n_anchors {cov2.n_anchors} → {n_max_line} "
            f"(span-limited by navmesh: span ≈ {D:.1f} m, min_sep ≈ {cov2.min_anchor_sep_m:.2f} m)"
        )
        cov2 = replace(cov2, n_anchors=n_max_line)

    return cov2, notes


def retarget_anchor_density(sim, cov, want=None, *, min_sep_floor=0.6, relax_iters=3):
    """
    Try to keep ≥ `want` anchors by gently relaxing spacing/clearance.

    On each iteration:
      • min_anchor_sep_m ← max(min_sep_floor, 0.8 × current)
      • anchor_min_clearance_m ← max(0.18, 0.8 × current)
      • attempt a tour; stop once anchors ≥ want

    Returns
      cov2  : possibly relaxed CoverageCfg
      notes : list[str] progress/success messages (no prints)
    """
    if want is None or want <= 0:
        return cov, ["no target anchor count requested"]

    notes = []
    cov2 = cov
    anchors = []
    for it in range(int(relax_iters)):
        # progressively relax
        prev_sep = cov2.min_anchor_sep_m
        prev_clr = cov2.anchor_min_clearance_m
        sep_new = max(min_sep_floor, 0.8 * prev_sep)
        clr_new = max(0.18, 0.8 * prev_clr)
        if sep_new == prev_sep and clr_new == prev_clr:
            break

        cov2 = replace(cov2, min_anchor_sep_m=sep_new, anchor_min_clearance_m=clr_new)

        from src.utils.habitat_traj import make_video_like_trajectory
        base_traj, anchors, _ = make_video_like_trajectory(
            sim,
            n_anchors=int(cov2.n_anchors),
            stride_m=float(cov2.traj_stride_m),
            seed=int(cov2.__dict__.get("seed", 0)),  # or pass render.seed from caller
            min_sep_geo=float(cov2.min_anchor_sep_m),
            closed_loop=True,
            densify_max_seg_m=0.20,
            smooth_passes=1,
            return_debug=True,
            anchor_min_clearance_m=float(cov2.anchor_min_clearance_m),
        )

        notes.append(
            f"relax[{it+1}/{relax_iters}]: min_sep {prev_sep:.2f}→{sep_new:.2f} m, "
            f"clearance {prev_clr:.2f}→{clr_new:.2f} m → anchors {len(anchors)}/{want}"
        )
        if len(anchors) >= want:
            notes.append(f"achieved {len(anchors)} anchors after {it+1} relax step(s)")
            return cov2, notes

    notes.append(f"still below target after relax (want={want}, got={len(anchors)})")
    return cov2, notes


def _geo_distance(pathfinder, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32)
    sp.requested_end   = np.asarray(b, np.float32)
    if pathfinder.find_path(sp) and np.isfinite(sp.geodesic_distance) and sp.geodesic_distance > 0:
        return float(sp.geodesic_distance)
    return np.inf


def _point_to_polyline_dist_xz(p, poly):
    """Min Euclidean XZ distance from point p to polyline poly (N,3)."""
    if poly is None or len(poly) < 2:
        return np.inf
    pxz = np.array([p[0], p[2]], np.float32)
    Q   = np.stack([poly[:,0], poly[:,2]], axis=1)  # (N,2)
    dmin = np.inf
    for i in range(len(Q)-1):
        a, b = Q[i], Q[i+1]
        ab = b - a
        t  = 0.0 if np.allclose(ab, 0) else np.clip(np.dot(pxz - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        proj = a + t * ab
        dmin = min(dmin, float(np.linalg.norm(pxz - proj)))
    return dmin


def seed_gap_anchors(
    sim,
    *,
    anchors: np.ndarray,
    path: np.ndarray = None,             # (Nv,3) visit/path (optional but recommended)
    min_radius_m: float = 1.0,           # require >= this geodesic distance to nearest anchor
    max_new: int = 8,                    # how many to add at most
    samples: int = 1500,                 # random navmesh probes
    min_clearance_m: float = 0.20,       # drop points with too little obstacle clearance
    near_path_band_m: float = 0.75,      # only keep candidates within this Euclidean XZ band of path
    euclid_gate_frac: float = 0.7,       # quick XZ pre-gate to save geodesic calls
) -> tuple:
    """
    Seed additional anchors in navmesh regions that are far from existing anchors.

    Keeps points that are:
      • geodesically ≥ min_radius_m from the current anchor set,
      • have obstacle clearance ≥ min_clearance_m,
      • (optional) lie within near_path_band_m of the current path (so refinement applies),
    then picks 'max_new' via greedy farthest-point sampling.

    Returns
    -------
    new_anchors : (K,3) float32 (K<=max_new)
    notes       : list[str] summary lines for logs
    """
    pf = sim.pathfinder
    A  = np.asarray(anchors, np.float32)
    P  = np.asarray(path, np.float32) if path is not None else None

    # Collect random navigable points
    pts = []
    for _ in range(int(samples)):
        q = np.asarray(pf.get_random_navigable_point(), np.float32).reshape(3)
        if not np.isfinite(q).all():
            continue
        # Optional: keep near the path so the nearest path index is meaningful
        if P is not None and near_path_band_m > 0:
            dpath = _point_to_polyline_dist_xz(q, P)
            if not np.isfinite(dpath) or dpath > float(near_path_band_m):
                continue
        # Clearance gate
        try:
            clr = float(pf.distance_to_closest_obstacle(q))
        except Exception:
            clr = np.inf  # if API missing, don't filter by clearance
        if not np.isfinite(clr) or clr < float(min_clearance_m):
            continue
        pts.append((q, clr))
    if not pts:
        return np.zeros((0,3), np.float32), ["gap-seed: no candidate points after clearance/path gating"]

    C = np.stack([p for p,_ in pts], axis=0)  # candidates (M,3)

    # Quick Euclidean-XZ gate before geodesic calls
    if A.size > 0:
        A_xz = A[:, [0,2]]
        C_xz = C[:, [0,2]]
        # min Euclid XZ distance to any anchor
        d_e = np.sqrt(((C_xz[:,None,:] - A_xz[None,:,:])**2).sum(-1)).min(axis=1)
        keep = d_e >= (euclid_gate_frac * float(min_radius_m))
        C = C[keep]
        if C.size == 0:
            return np.zeros((0,3), np.float32), ["gap-seed: all candidates are too close (Euclid pre-gate)"]

    # Compute geodesic distance to nearest anchor
    dmin = []
    if A.size == 0:
        # No anchors yet → everyone is "far"
        dmin = np.full((C.shape[0],), np.inf, np.float32)
    else:
        for q in C:
            dnear = np.inf
            for a in A:
                dnear = min(dnear, _geo_distance(pf, q, a))
                if dnear < float(min_radius_m) * 0.5:
                    break  # early out
            dmin.append(dnear)
        dmin = np.asarray(dmin, np.float32)

    # Keep only those that satisfy the geodesic radius
    keep = dmin >= float(min_radius_m)
    C    = C[keep]
    dmin = dmin[keep]
    if C.size == 0:
        return np.zeros((0,3), np.float32), [f"gap-seed: no points ≥ {min_radius_m:.2f} m from anchors (geodesic)"]

    # Greedy FPS over geodesic distance (update dmin after each selection)
    new_pts = []
    notes   = []
    for _ in range(int(max_new)):
        i = int(np.argmax(dmin))
        if not np.isfinite(dmin[i]) or dmin[i] < float(min_radius_m):
            break
        q_star = C[i]
        new_pts.append(q_star)

        # Update dmin with distances to the newly added anchor
        d_to_new = np.array([_geo_distance(pf, q, q_star) for q in C], np.float32)
        dmin = np.minimum(dmin, d_to_new)

    if not new_pts:
        notes.append("gap-seed: nothing added (candidates failed final radius check)")
        return np.zeros((0,3), np.float32), notes

    NA = np.stack(new_pts, 0).astype(np.float32)
    notes.append(
        f"gap-seed: added {len(new_pts)} anchor(s) "
        f"(min geodesic radius ≥ {min_radius_m:.2f} m, near-path band ≤ {near_path_band_m:.2f} m)"
    )
    return NA, notes


# --- relaxed wrapper with debug ------------------------------------------------
def seed_gap_anchors_relaxed(
    sim,
    *,
    anchors,
    path=None,
    base_min_radius_m: float,
    max_new: int,
    min_clearance_m: float,
    near_path_band_m: float = 0.75,
    samples: int = 2000,
    radii_factors=(1.0, 0.8, 0.6, 0.5),
    euclid_gate_frac: float = 0.7,
):
    """
    Try gap-seeding with a decreasing radius schedule until something sticks.
    Returns (extra_anchors[K,3], notes[list[str]]).
    """
    notes_all = []
    # tiny inner helper to print candidate counts from seed_gap_anchors
    def _with_counts(sim, **kw):
        extra, notes = seed_gap_anchors(sim, **kw)
        return extra, notes

    for f in radii_factors:
        r = max(0.4, float(base_min_radius_m) * float(f))
        extra, notes = _with_counts(
            sim,
            anchors=np.asarray(anchors, np.float32),
            path=np.asarray(path, np.float32) if path is not None else None,
            min_radius_m=r,
            max_new=int(max_new),
            samples=int(samples),
            min_clearance_m=float(min_clearance_m),
            near_path_band_m=float(near_path_band_m),
            euclid_gate_frac=float(euclid_gate_frac),
        )
        notes_all += [f"gap-seed try r={r:.2f} m → {len(extra)} added"] + notes
        if len(extra) > 0:
            return extra, notes_all

    return np.zeros((0,3), np.float32), notes_all + ["gap-seed: nothing added after relaxed schedule"]


def _geo_distance(pathfinder, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, np.float32)
    sp.requested_end   = np.asarray(b, np.float32)
    ok = pathfinder.find_path(sp)
    if ok and np.isfinite(sp.geodesic_distance) and sp.geodesic_distance > 0:
        return float(sp.geodesic_distance)
    return np.inf

def as_np_point_list(pts) -> list:
    """
    Convert points to [np.ndarray([x,y,z], float32), ...].

    Accepts:
      - (N,3) ndarray
      - iterable of 3D points (lists/tuples/arrays)

    Returns:
      list[np.ndarray shape (3,), dtype float32]
    """
    P = np.asarray(pts, dtype=np.float32)
    if P.ndim == 1:
        if P.size == 0:
            return []
        P = P.reshape(1, 3)
    else:
        P = P.reshape(-1, 3)
    # Return independent, contiguous 1D vectors
    return [np.ascontiguousarray(p.copy()) for p in P]


def seed_farthest_on_navmesh(
    sim,
    *,
    anchors: np.ndarray,
    max_new: int,
    min_radius_m: float,
    min_clearance_m: float = 0.18,
    samples: int = 8000,
    batch: int = 2500,
    rng_seed: int = 0,
    restrict_to_anchor_island: bool = True,   # NEW: keep candidates on same island
    return_np_list: bool = True,
):
    """
    Greedy farthest-point seeding over the navmesh.

    If `restrict_to_anchor_island=True`, a candidate must be geodesically
    connected to the first anchor; otherwise it's skipped. This avoids adding
    points from disconnected navmesh islands that would always yield inf distances.
    """
    pf = sim.pathfinder
    rng = np.random.default_rng(rng_seed)

    A = np.asarray(anchors, np.float32)
    if A.size == 0:
        return np.zeros((0, 3), np.float32), ["no anchors to seed around"]

    root = A[0]  # reference island

    extras, notes = [], []
    def _clearance_ok(p):
        try:
            c = pf.distance_to_closest_obstacle(p.astype(np.float32))
            return np.isfinite(c) and (c >= float(min_clearance_m))
        except Exception:
            return True  # if API not available, skip this gate

    def _connected_to_root(p):
        if not restrict_to_anchor_island:
            return True
        d = _geo_distance(pf, p, root)
        return np.isfinite(d)

    def _min_geo_to_set(p):
        best = np.inf
        for q in A:
            d = _geo_distance(pf, p, q)
            if d < best: best = d
        for q in extras:
            d = _geo_distance(pf, p, q)
            if d < best: best = d
        return best

    tried = 0
    stalls = 0
    while len(extras) < int(max_new) and tried < max(3 * samples, 6000):
        # draw a batch of candidates anywhere on navmesh
        C = []
        for _ in range(batch):
            p = np.asarray(pf.get_random_navigable_point(), np.float32).reshape(3)
            if np.isfinite(p).all() and _clearance_ok(p) and _connected_to_root(p):
                C.append(p)
        tried += batch
        if not C:
            notes.append("farthest: batch had 0 candidates on anchor island (or clearance failed)")
            stalls += 1
            # gentle radius drop if we’re stalling
            if stalls % 2 == 0:
                min_radius_m = max(0.4, 0.9 * float(min_radius_m))
                notes.append(f"farthest: relax radius → {min_radius_m:.2f} m")
            continue

        C = np.stack(C, 0)
        mind = np.empty((len(C),), np.float32)
        all_inf = True
        for i, p in enumerate(C):
            dmin = _min_geo_to_set(p)
            mind[i] = dmin
            if np.isfinite(dmin):
                all_inf = False

        if all_inf:
            notes.append("farthest: all batch candidates were disconnected (min-geo = inf); skipping batch")
            stalls += 1
            if stalls % 2 == 0:
                min_radius_m = max(0.4, 0.9 * float(min_radius_m))
                notes.append(f"farthest: relax radius → {min_radius_m:.2f} m")
            continue

        j = int(np.argmax(mind))
        best = float(mind[j])

        if np.isfinite(best) and best >= float(min_radius_m):
            extras.append(C[j])
            notes.append(f"farthest: +1 anchor (min-geo ≈ {best:.2f} m, extras={len(extras)}/{max_new})")
        else:
            reason = "not finite" if not np.isfinite(best) else f"< radius {min_radius_m:.2f} m"
            notes.append(f"farthest: best candidate min-geo {best:.2f} m ({reason}); no add")

    extras_np = np.stack(extras, 0).astype(np.float32) if extras else np.zeros((0, 3), np.float32)
    if return_np_list:
        return as_np_point_list(extras_np), notes
    return extras_np, notes

    # return (np.stack(extras, 0).astype(np.float32) if extras else np.zeros((0, 3), np.float32), notes)