import cv2
import matplotlib
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import re, numpy as np
import imageio.v3 as iio

def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def colorize_depth(
    depth: np.ndarray,
    mask: np.ndarray = None,
    normalize: bool = True,
    cmap: str = 'Spectral'
) -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)

    disp = 1.0 / depth

    if normalize:
        lo, hi = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - lo) / (hi - lo + 1e-12)

    # Forward-compatible colormap getter
    try:
        import matplotlib
        cmap_fn = matplotlib.colormaps[cmap]  # matplotlib >= 3.8
    except AttributeError:
        import matplotlib.cm
        cmap_fn = matplotlib.cm.get_cmap(cmap)  # matplotlib <= 3.7

    colored = np.nan_to_num(cmap_fn(1.0 - disp)[..., :3], 0)
    return np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))

def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5   # visualize
    return (normal.clip(0, 1) * 255).astype(np.uint8)

def save_ply(path, xyz, rgb=None):
    if rgb is None: rgb = np.zeros_like(xyz, dtype=np.uint8)
    verts = np.zeros(xyz.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),
                                          ('red','u1'),('green','u1'),('blue','u1')])
    verts['x'],verts['y'],verts['z'] = xyz[:,0],xyz[:,1],xyz[:,2]
    verts['red'],verts['green'],verts['blue'] = rgb[:,0],rgb[:,1],rgb[:,2]
    PlyData([PlyElement.describe(verts, 'vertex')]).write(str(path))

def read_ply_xyzrgb(path):
    """Read a PLY with x,y,z and optional RGB into (xyz, rgb)."""
    pd = PlyData.read(str(path))
    v = pd['vertex'].data
    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    if {'red','green','blue'}.issubset(v.dtype.names):
        rgb = np.vstack([v['red'], v['green'], v['blue']]).T.astype(np.uint8)
    else:
        rgb = np.zeros_like(xyz, dtype=np.uint8)
    return xyz, rgb

def _pack_xyz_int(keys: np.ndarray) -> np.ndarray:
    """
    Pack integer voxel coordinates (N,3) into uint64:
      key = (x<<42) | (y<<21) | z, assuming each axis fits in 21 bits (0..2,097,151).
    If any axis exceeds that, raise and let caller fall back to another method.
    """
    if keys.dtype != np.int64:
        keys = keys.astype(np.int64, copy=False)
    mins = keys.min(axis=0)
    k = keys - mins[None, :]                     # shift to non-negative
    if (k.max(axis=0) >= (1 << 21)).any():
        raise OverflowError("Voxel indices exceed 21 bits; increase voxel_size or use method='unique'.")
    return ((k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]).astype(np.uint64), mins

def voxel_merge(points_list,
                colors_list,
                voxel_size: float = 0.015,
                min_pts_per_voxel: int = 3,
                method: str = "unique"):
    """
    Voxel-grid merge (downsample) a list of per-view point clouds.

    Parameters
    ----------
    points_list : list[np.ndarray]
        List of (Ni,3) float32 arrays in the **same world frame** (meters).
    colors_list : list[np.ndarray]
        List of (Ni,3) uint8 arrays (RGB) aligned with points_list.
    voxel_size : float
        Edge length of cubic voxels in meters (e.g., 0.01 = 1cm).
    min_pts_per_voxel : int
        Drop voxels supported by fewer than this many input points.
    method : {"unique","packed"}
        Backend:
          - "unique": quantize → `np.unique(axis=0)` → `np.bincount` reductions.
          - "packed": quantize → pack 3D keys to 1D uint64 → sort + `reduceat`.
            Usually faster and lower-memory for large clouds.

    Returns
    -------
    xyz_out : (M,3) float32
        Voxel-averaged 3D points (centroids of points inside each voxel).
    rgb_out : (M,3) uint8
        Corresponding per-voxel mean color (rounded).
    """
    if len(points_list) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    pts = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
    cols = np.concatenate(colors_list, axis=0).astype(np.float32, copy=False)  # accumulate in float
    v = float(voxel_size)

    # Integer voxel indices
    keys_3d = np.floor(pts / v).astype(np.int64, copy=False)

    if method == "packed":
        try:
            codes, mins = _pack_xyz_int(keys_3d)
            order = np.argsort(codes)
            codes_sorted = codes[order]
            pts_sorted = pts[order]
            cols_sorted = cols[order]

            # segment boundaries
            # indices where code changes
            edge = np.flatnonzero(np.concatenate(
                [np.array([True]), codes_sorted[1:] != codes_sorted[:-1]]
            ))
            counts = np.diff(np.concatenate([edge, np.array([len(codes_sorted)])]))

            # reduce sums per segment
            xyz_sum = np.add.reduceat(pts_sorted, edge, axis=0)
            rgb_sum = np.add.reduceat(cols_sorted, edge, axis=0)

            keep = counts >= int(min_pts_per_voxel)
            if not np.any(keep):
                return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

            xyz_mean = (xyz_sum[keep] / counts[keep, None]).astype(np.float32)
            rgb_mean = (rgb_sum[keep] / counts[keep, None])

            # (Optional) snap to voxel centers for a regularized cloud:
            # centers = (np.floor(xyz_mean / v) * v + v*0.5).astype(np.float32)

            rgb_out = np.clip(np.round(rgb_mean), 0, 255).astype(np.uint8)
            return xyz_mean, rgb_out

        except OverflowError:
            # fall back to "unique"
            method = "unique"

    # --- method == "unique" ---
    labels, inverse, counts = np.unique(keys_3d, axis=0, return_inverse=True, return_counts=True)
    L = labels.shape[0]
    sum_x = np.bincount(inverse, weights=pts[:, 0], minlength=L)
    sum_y = np.bincount(inverse, weights=pts[:, 1], minlength=L)
    sum_z = np.bincount(inverse, weights=pts[:, 2], minlength=L)
    sum_r = np.bincount(inverse, weights=cols[:, 0], minlength=L)
    sum_g = np.bincount(inverse, weights=cols[:, 1], minlength=L)
    sum_b = np.bincount(inverse, weights=cols[:, 2], minlength=L)

    keep = counts >= int(min_pts_per_voxel)
    if not np.any(keep):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    denom = np.maximum(counts[keep], 1)[:, None]
    xyz_out = np.stack([sum_x[keep], sum_y[keep], sum_z[keep]], axis=1) / denom
    rgb_out = np.stack([sum_r[keep], sum_g[keep], sum_b[keep]], axis=1) / denom
    return xyz_out.astype(np.float32), np.clip(np.round(rgb_out), 0, 255).astype(np.uint8)

def merge_per_view_pointclouds(ply_dir,
                               out_ply,
                               voxel_size: float = 0.015,
                               min_pts_per_voxel: int = 3,
                               method: str = "packed"):
    """
    Read all per-view PLYs from `ply_dir`, merge them in world coordinates with
    voxel averaging, and write a single PLY to `out_ply`.

    Parameters
    ----------
    ply_dir : str | Path
        Directory containing `points_*.ply` files (each with x,y,z and RGB).
    out_ply : str | Path
        Destination PLY path (will be created).
    voxel_size : float
        Voxel edge length in meters (e.g., 0.015 for ~1.5cm).
    min_pts_per_voxel : int
        Discard voxels with fewer input points than this.
    method : {"packed","unique"}
        Merge backend; see `voxel_merge` for details.

    Notes
    -----
    - This assumes all per-view clouds are already in the same world frame.
    - For massive datasets: consider streaming (update accumulators per file),
      or save per-view `.npz` (xyz/rgb) to speed up I/O.
    """
    ply_dir = Path(ply_dir)
    xs, cs = [], []
    for p in sorted(ply_dir.glob("points_*.ply")):
        xyz, rgb = read_ply_xyzrgb(p)
        if xyz.size:
            xs.append(xyz); cs.append(rgb)
    if len(xs) == 0:
        print("[merge] no per-view PLYs found.")
        return

    xyz, rgb = voxel_merge(xs, cs,
                           voxel_size=float(voxel_size),
                           min_pts_per_voxel=int(min_pts_per_voxel),
                           method=method)

    # reuse your existing save_ply
    from .vis import save_ply
    Path(out_ply).parent.mkdir(parents=True, exist_ok=True)
    save_ply(out_ply, xyz, rgb)
    print(f"[merge] {len(xs)} PLYs → {xyz.shape[0]} points @ {voxel_size*100:.1f} cm → {out_ply}")

def merge_per_view_pointclouds_fast(
    exr_dir,
    rgb_dir,
    out_ply,
    *,
    voxel_size: float = 0.015,
    min_pts_per_voxel: int = 3,
    device: str = "cuda",
    method: str = "packed",       # "packed" (fast) or "unique" (slower, no 64-bit packing)
    max_views = None, # cap for debugging
    log_every: int = 10,
):
    """
    Merge per-view EXR world points (H,W,3) with matching RGB images into one voxelized cloud.

    - Per view:
        * Load points_EXR: pts_w (H,W,3) float32 in world coords
        * Load RGB: rgb (H,W,3) uint8
        * Mask invalid by finite(x,y,z)
        * Quantize by voxel_size; pack to 64-bit keys (if 'packed') on GPU
        * Sort + unique_consecutive; segment-reduce xyz/rgb sums and counts (GPU)
        * Ship per-view aggregates to CPU

    - Final:
        * Concatenate all per-view aggregates (CPU), sort by key, reduce with numpy
        * Filter by min_pts_per_voxel
        * Save voxel-mean xyz/rgb to out_ply

    Notes
    -----
    • 'packed' uses 21 bits per axis with a fixed SHIFT=2^20, i.e. indices must lie in [-2^20, 2^20-1].
      If exceeded, raise with a helpful message (increase voxel_size or switch to method='unique').
    • device='cuda' recommended; falls back to CPU if no CUDA.
    """
    if not _torch_available():
        raise ImportError("merge_per_view_pointclouds_fast requires PyTorch installed.")
    import torch

    exr_dir = Path(exr_dir)
    rgb_dir = Path(rgb_dir)
    out_ply = Path(out_ply)

    exr_paths = sorted(exr_dir.glob("points_*.exr"))
    if max_views is not None:
        exr_paths = exr_paths[:max_views]
    if len(exr_paths) == 0:
        print(f"[fast-merge] no EXRs under {exr_dir}")
        return

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    device_t = torch.device("cuda" if use_cuda else "cpu")
    SHIFT = 1 << 20  # 1,048,576; 21-bit safety window per axis
    v = float(voxel_size)

    # per-view aggregates we’ll merge at the end (CPU numpy)
    all_codes: list[np.ndarray] = []
    all_xyz_sums: list[np.ndarray] = []
    all_rgb_sums: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []

    r_idx = re.compile(r"points_(\d+)\.exr$")

    for vi, exr_path in enumerate(exr_paths):
        m = r_idx.search(exr_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        rgb_path = rgb_dir / f"rgb_{idx:05d}.png"

        # --- load points (EXR) ---
        pts = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)  # (H,W,3) float32
        if pts is None or pts.ndim != 3 or pts.shape[2] < 3:
            continue
        H, W, _ = pts.shape
        xyz = pts[:, :, :3].reshape(-1, 3).astype(np.float32, copy=False)

        # --- mask invalid ---
        valid = np.isfinite(xyz).all(axis=1)
        if not np.any(valid):
            continue
        xyz = xyz[valid]

        # --- load color (PNG) ---
        if rgb_path.exists():
            rgb = iio.imread(rgb_path)
            if rgb.ndim == 3 and rgb.shape[2] >= 3:
                rgb = rgb[:, :, :3]
            else:
                rgb = np.zeros((H, W, 3), np.uint8)
        else:
            rgb = np.zeros((H, W, 3), np.uint8)
        rgb = rgb.reshape(-1, 3)[valid].astype(np.float32, copy=False)

        # --- to torch ---
        xyz_t = torch.from_numpy(xyz).to(device_t, dtype=torch.float32)
        rgb_t = torch.from_numpy(rgb).to(device_t, dtype=torch.float32)

        # quantize to voxel indices
        q = torch.floor(xyz_t / v).to(torch.int64)

        # choose method
        local_method = method
        if local_method == "packed":
            # ensure indices fit into [0, 2^21)
            q_shift = q + SHIFT
            # quick range check on device
            qmin = torch.min(q_shift)
            qmax = torch.max(q_shift)
            if (qmin < 0) or (qmax >= (1 << 21)):
                raise OverflowError(
                    f"[fast-merge] voxel index overflow for {exr_path.name}: "
                    f"min={int(qmin.item())}, max={int(qmax.item())}. "
                    f"Increase voxel_size (current={voxel_size}m) OR set method='unique'."
                )
            codes = (q_shift[:, 0] << 42) | (q_shift[:, 1] << 21) | q_shift[:, 2]
        else:
            # fallback: "unique" (no packing). We'll still pack later if possible.
            codes = None

        if local_method == "packed":
            # sort + unique_consecutive
            order = torch.argsort(codes)
            codes_sorted = codes[order]
            xyz_s = xyz_t[order]
            rgb_s = rgb_t[order]

            uniq, inverse, counts = torch.unique_consecutive(
                codes_sorted, return_inverse=True, return_counts=True
            )
            K = int(uniq.shape[0])
            idx_expand = inverse[:, None].expand(-1, 3)

            xyz_sum = torch.zeros((K, 3), dtype=torch.float64, device=device_t)
            rgb_sum = torch.zeros((K, 3), dtype=torch.float64, device=device_t)
            xyz_sum.scatter_add_(0, idx_expand, xyz_s.to(torch.float64))
            rgb_sum.scatter_add_(0, idx_expand, rgb_s.to(torch.float64))

            # ship to CPU numpy
            all_codes.append(uniq.detach().cpu().numpy().astype(np.uint64))
            all_xyz_sums.append(xyz_sum.detach().cpu().numpy())
            all_rgb_sums.append(rgb_sum.detach().cpu().numpy())
            all_counts.append(counts.detach().cpu().numpy())

        else:
            # method == "unique": unique rows on q (potentially slower), then try to pack uniques
            uniq, inverse, counts = torch.unique(q, dim=0, return_inverse=True, return_counts=True)
            K = int(uniq.shape[0])
            idx_expand = inverse[:, None].expand(-1, 3)

            xyz_sum = torch.zeros((K, 3), dtype=torch.float64, device=device_t)
            rgb_sum = torch.zeros((K, 3), dtype=torch.float64, device=device_t)
            xyz_sum.scatter_add_(0, idx_expand, xyz_t.to(torch.float64))
            rgb_sum.scatter_add_(0, idx_expand, rgb_t.to(torch.float64))

            # attempt to pack uniques with SHIFT (faster final reduce)
            u = uniq.to(torch.int64)
            u_shift = u + SHIFT
            if (torch.min(u_shift) < 0) or (torch.max(u_shift) >= (1 << 21)):
                # can't pack safely; keep raw 3D integer labels (slower final reduce)
                all_codes.append(u.detach().cpu().numpy().astype(np.int64))   # shape (K,3)
            else:
                packed = ((u_shift[:, 0] << 42) | (u_shift[:, 1] << 21) | u_shift[:, 2]).to(torch.int64)
                all_codes.append(packed.detach().cpu().numpy().astype(np.uint64))  # shape (K,)

            all_xyz_sums.append(xyz_sum.detach().cpu().numpy())
            all_rgb_sums.append(rgb_sum.detach().cpu().numpy())
            all_counts.append(counts.detach().cpu().numpy())

        if (vi + 1) % max(1, log_every) == 0:
            print(f"[fast-merge] pre-aggregated {vi+1}/{len(exr_paths)} views")

    if len(all_codes) == 0:
        print("[fast-merge] nothing to merge.")
        return

    # --- final CPU merge ---
    # If any chunk produced 3D-label codes (shape (K,3)), do a slower row-merge; else do packed fast merge.
    any_row_labels = any(c.ndim == 2 for c in all_codes)

    if any_row_labels:
        # stack 3D integer labels; convert packed (1D) chunks to 3D by unpacking (inverse pack)
        # NOTE: we only expect this when method='unique' and pack failed; to keep code compact,
        # we simply merge 3D labels with numpy.unique.
        labels_list = []
        for c in all_codes:
            if c.ndim == 1:
                # unpack 64-bit back to 3x int (inverse of (x<<42)|(y<<21)|z with SHIFT)
                x = (c >> 42) & ((1 << 21) - 1)
                y = (c >> 21) & ((1 << 21) - 1)
                z = c & ((1 << 21) - 1)
                labels_list.append(np.stack([x - SHIFT, y - SHIFT, z - SHIFT], axis=1).astype(np.int64))
            else:
                labels_list.append(c)
        labels = np.concatenate(labels_list, axis=0)
        xyz_sums = np.concatenate(all_xyz_sums, axis=0)
        rgb_sums = np.concatenate(all_rgb_sums, axis=0)
        counts  = np.concatenate(all_counts, axis=0)

        lab_unique, inverse = np.unique(labels, axis=0, return_inverse=True)
        L = lab_unique.shape[0]
        # reduce by inverse
        sum_x = np.bincount(inverse, weights=xyz_sums[:, 0], minlength=L)
        sum_y = np.bincount(inverse, weights=xyz_sums[:, 1], minlength=L)
        sum_z = np.bincount(inverse, weights=xyz_sums[:, 2], minlength=L)
        sum_r = np.bincount(inverse, weights=rgb_sums[:, 0], minlength=L)
        sum_g = np.bincount(inverse, weights=rgb_sums[:, 1], minlength=L)
        sum_b = np.bincount(inverse, weights=rgb_sums[:, 2], minlength=L)
        cnt   = np.bincount(inverse, weights=counts.astype(np.int64), minlength=L)

        keep = cnt >= int(min_pts_per_voxel)
        if not np.any(keep):
            print("[fast-merge] no voxels survive min_pts_per_voxel.")
            return

        denom = np.maximum(cnt[keep], 1)[:, None]
        xyz_out = np.stack([sum_x[keep], sum_y[keep], sum_z[keep]], axis=1) / denom
        rgb_out = np.stack([sum_r[keep], sum_g[keep], sum_b[keep]], axis=1) / denom
        xyz_out = xyz_out.astype(np.float32)
        rgb_out = np.clip(np.round(rgb_out), 0, 255).astype(np.uint8)

    else:
        # all are packed uint64 → fast sort + reduce
        codes = np.concatenate(all_codes, axis=0).astype(np.uint64, copy=False)
        xyz_sums = np.concatenate(all_xyz_sums, axis=0)
        rgb_sums = np.concatenate(all_rgb_sums, axis=0)
        counts   = np.concatenate(all_counts, axis=0)

        order = np.argsort(codes, kind="mergesort")
        codes_s = codes[order]
        xyz_s = xyz_sums[order]
        rgb_s = rgb_sums[order]
        cnt_s = counts[order]

        edge = np.flatnonzero(np.concatenate([np.array([True]), codes_s[1:] != codes_s[:-1]]))
        cnt_vox = np.diff(np.concatenate([edge, np.array([len(codes_s)])]))

        xyz_sum = np.add.reduceat(xyz_s, edge, axis=0)
        rgb_sum = np.add.reduceat(rgb_s, edge, axis=0)
        cnt_sum = np.add.reduceat(cnt_s, edge, axis=0)

        keep = cnt_sum >= int(min_pts_per_voxel)
        if not np.any(keep):
            print("[fast-merge] no voxels survive min_pts_per_voxel.")
            return

        xyz_out = (xyz_sum[keep] / cnt_sum[keep, None]).astype(np.float32)
        rgb_out = np.clip(np.round(rgb_sum[keep] / cnt_sum[keep, None]), 0, 255).astype(np.uint8)

    # write
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    save_ply(out_ply, xyz_out, rgb_out)
    print(f"[fast-merge] wrote {xyz_out.shape[0]} voxels @ {voxel_size*100:.1f} cm → {out_ply}")