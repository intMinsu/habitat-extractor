# habitat_traj_vis.py
import numpy as np
import math
from pathlib import Path
from plyfile import PlyData, PlyElement
import cv2
import trimesh
from PIL import Image
from trimesh.visual import texture as ttex

def _load_vertices_rgb_plyfile(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    def _get(name, default=None, dtype=np.float32):
        if name in v.data.dtype.names:
            return np.asarray(v[name], dtype=dtype)
        return None if default is None else np.full((len(v),), default, dtype=dtype)
    x = _get("x"); y = _get("y"); z = _get("z")
    r = _get("red",   200, np.float32)
    g = _get("green", 200, np.float32)
    b = _get("blue",  200, np.float32)
    return np.stack([x, y, z, r, g, b], axis=1).astype(np.float32)

def _write_vertices_only_plyfile(path: Path, verts_rgb: np.ndarray, *, ascii: bool = False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    vrec = np.empty(
        verts_rgb.shape[0],
        dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")]
    )
    vrec["x"] = verts_rgb[:,0]; vrec["y"] = verts_rgb[:,1]; vrec["z"] = verts_rgb[:,2]
    vrec["red"]   = np.clip(verts_rgb[:,3], 0, 255).astype(np.uint8)
    vrec["green"] = np.clip(verts_rgb[:,4], 0, 255).astype(np.uint8)
    vrec["blue"]  = np.clip(verts_rgb[:,5], 0, 255).astype(np.uint8)
    PlyData([PlyElement.describe(vrec, "vertex")], text=ascii).write(str(path))

def _pixel_to_cam(u, v, depth, fx, fy, cx, cy):
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    return np.array([x, y, z], dtype=np.float32)

def _apply_T(pts_c: np.ndarray, T_c2w: np.ndarray) -> np.ndarray:
    R = T_c2w[:3, :3]; t = T_c2w[:3, 3]
    return pts_c @ R.T + t

def _frustum_corners_world(T_c2w: np.ndarray, intr: dict, near: float, far: float) -> np.ndarray:
    W = int(intr["W"]); H = int(intr["H"])
    fx = float(intr["fx"]); fy = float(intr["fy"]); cx = float(intr["cx"]); cy = float(intr["cy"])
    tl = (0.0, 0.0); tr = (W-1.0, 0.0); br = (W-1.0, H-1.0); bl = (0.0, H-1.0)
    n_tl = _pixel_to_cam(*tl, near, fx, fy, cx, cy)
    n_tr = _pixel_to_cam(*tr, near, fx, fy, cx, cy)
    n_br = _pixel_to_cam(*br, near, fx, fy, cx, cy)
    n_bl = _pixel_to_cam(*bl, near, fx, fy, cx, cy)
    f_tl = _pixel_to_cam(*tl, far,  fx, fy, cx, cy)
    f_tr = _pixel_to_cam(*tr, far,  fx, fy, cx, cy)
    f_br = _pixel_to_cam(*br, far,  fx, fy, cx, cy)
    f_bl = _pixel_to_cam(*bl, far,  fx, fy, cx, cy)
    C = np.stack([n_tl, n_tr, n_br, n_bl, f_tl, f_tr, f_br, f_bl], axis=0)
    return _apply_T(C, T_c2w)  # (8,3) in world

def _sample_segment(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(2, int(n)), dtype=np.float32)[:, None]
    return (1.0 - t) * p0[None, :] + t * p1[None, :]

def _axes_tripod_points(T_c2w: np.ndarray, axis_len: float, samples: int = 6) -> np.ndarray:
    """Return points along +X (red), +Y (green), +Z (blue) rays from camera center."""
    R = T_c2w[:3, :3]; t = T_c2w[:3, 3]
    axes = [R[:,0]*axis_len, R[:,1]*axis_len, R[:,2]*axis_len]
    cols = [(255,0,0), (0,255,0), (0,0,255)]
    out = []
    for d, c in zip(axes, cols):
        seg = _sample_segment(t, t + d, samples)
        rgb = np.tile(np.array(c, np.float32)[None, :], (seg.shape[0], 1))
        out.append(np.concatenate([seg, rgb], axis=1))
    return np.concatenate(out, axis=0).astype(np.float32)

def overlay_pose_frusta_as_points_on_pointcloud(
    in_ply: Path,
    out_ply: Path,
    poses_meta: list,
    intr: dict,
    *,
    near: float = 0.06,
    far: float = 0.22,
    edge_samples: int = 12,
    every: int = 1,
    add_axes: bool = True,
    axis_len: float = 0.10,
    frustum_color=(10,10,10),
    ascii: bool = False,
):
    """
    Create a vertex-only PLY that contains:
      - all base point-cloud vertices, PLUS
      - frustum edges sampled as points, PLUS (optional)
      - tiny RGB axis tripods at each camera.

    This avoids the 'unreferenced vertex pruning' problem in viewers.
    """
    base = _load_vertices_rgb_plyfile(in_ply)  # (Nb,6)
    pts = [base]

    step = max(1, int(every))
    for m in poses_meta[::step]:
        T = np.asarray(m["T_c2w"], np.float32)
        corners = _frustum_corners_world(T, intr, near, far)
        # indices: n0..n3, f0..f3
        n0,n1,n2,n3,f0,f1,f2,f3 = range(8)
        edges = [
            (n0,n1),(n1,n2),(n2,n3),(n3,n0),  # near
            (f0,f1),(f1,f2),(f2,f3),(f3,f0),  # far
            (n0,f0),(n1,f1),(n2,f2),(n3,f3),  # pillars
        ]
        color = np.array(frustum_color, np.float32)
        for a,b in edges:
            seg = _sample_segment(corners[a], corners[b], edge_samples)
            rgb = np.tile(color[None, :], (seg.shape[0], 1))
            pts.append(np.concatenate([seg, rgb], axis=1))
        if add_axes:
            pts.append(_axes_tripod_points(T, axis_len, samples=max(3, edge_samples//2)))
    all_pts = np.concatenate(pts, axis=0).astype(np.float32)
    _write_vertices_only_plyfile(out_ply, all_pts, ascii=ascii)


def _minmax(v):
    return float(v.min()), float(v.max())

def save_birdeye_poses_png(out_png: Path,
                           poses_meta: list,
                           *,
                           anchors = None,
                           visit = None,
                           px: int = 1024,
                           frustum_len_m: float = 0.45,
                           frustum_width_m: float = 0.28,
                           every: int = 1):
    """
    Top-down (X,Z) view with small wedge-shaped frusta drawn from poses.
    - +Z points upward on the image.
    - Heading is read from T_c2w (camera +Z mapped to world).
    """
    if poses_meta is None or len(poses_meta) == 0:
        return

    P = np.array([m["position_world"] for m in poses_meta], np.float32)  # (N,3)
    xs = [P[:,0]]; zs = [P[:,2]]
    if anchors is not None and anchors.size: xs.append(anchors[:,0]); zs.append(anchors[:,2])
    if visit   is not None and visit.size:   xs.append(visit[:,0]);   zs.append(visit[:,2])

    xmin, xmax = _minmax(np.concatenate(xs))
    zmin, zmax = _minmax(np.concatenate(zs))
    pad = 0.06 * max(xmax - xmin, zmax - zmin, 1.0)
    xmin -= pad; xmax += pad; zmin -= pad; zmax += pad

    W = px
    H = int(round(px * (zmax - zmin) / max(1e-6, (xmax - xmin))))
    img = np.ones((H, W, 3), np.uint8) * 255

    def to_px(x, z):
        u = (x - xmin) / max(1e-6, (xmax - xmin))
        v = (z - zmin) / max(1e-6, (zmax - zmin))
        return int(round(u * (W - 1))), int(round((1.0 - v) * (H - 1)))  # +Z up

    # visit path (blue-ish)
    if visit is not None and visit.size:
        for i in range(1, visit.shape[0]):
            cv2.line(img, to_px(visit[i-1,0], visit[i-1,2]),
                          to_px(visit[i,0],   visit[i,2]), (200, 80, 0), 2)

    # anchors (red)
    if anchors is not None and anchors.size:
        for a in anchors:
            cv2.circle(img, to_px(a[0], a[2]), 5, (0,0,255), -1)

    # frusta wedges for poses (dark)
    step = max(1, int(every))
    for k in range(0, len(poses_meta), step):
        m = poses_meta[k]
        T = np.asarray(m["T_c2w"], np.float32)
        R = T[:3,:3]; t = T[:3,3]
        fwd = R[:,2]  # camera +Z in world
        fwd_xz = np.array([fwd[0], fwd[2]], np.float32)
        nrm = np.linalg.norm(fwd_xz)
        if nrm < 1e-6:
            continue
        d = fwd_xz / nrm

        # 2D wedge points in world XZ
        p0 = np.array([t[0], t[2]], np.float32)
        tip = p0 + d * frustum_len_m
        ctr = p0 + d * (0.60 * frustum_len_m)
        left = ctr + np.array([-d[1], d[0]], np.float32) * (0.5 * frustum_width_m)
        right = ctr + np.array([ d[1],-d[0]], np.float32) * (0.5 * frustum_width_m)

        pts = np.array([to_px(*p0), to_px(*left), to_px(*tip), to_px(*right)], np.int32)
        cv2.fillConvexPoly(img, pts, (30,30,30))
        cv2.polylines(img, [pts], isClosed=True, color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)

    # compass
    base = np.array([xmin + 0.08*(xmax-xmin), zmin + 0.08*(zmax-zmin)], np.float32)
    u0 = to_px(*base); ux = to_px(base[0]+0.5, base[1]); uz = to_px(base[0], base[1]+0.5)
    cv2.arrowedLine(img, u0, ux, (0,0,0), 2, tipLength=0.25); cv2.putText(img, "+X", (ux[0]+4, ux[1]), 0, 0.45, (0,0,0), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, u0, uz, (0,0,0), 2, tipLength=0.25); cv2.putText(img, "+Z", (uz[0]+4, uz[1]), 0, 0.45, (0,0,0), 1, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)


def _plyfile_load_xyz_rgb(path: Path) -> tuple:
    """
    Read vertices (x,y,z) and per-vertex colors (r,g,b) from a PLY via plyfile.
    Returns (V[N,3], C[N,3]) as float32 in world units and 0..255 colors.
    Missing color channels default to mid-gray.
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    def get(name, default=None, dtype=np.float32):
        if name in v.data.dtype.names:
            return np.asarray(v[name], dtype=dtype)
        return np.full((len(v),), default, dtype=dtype) if default is not None else None
    V = np.stack([get("x"), get("y"), get("z")], axis=1).astype(np.float32)
    R = get("red",   200); G = get("green", 200); B = get("blue", 200)
    C = np.stack([R, G, B], axis=1).astype(np.float32)
    return V, C

def _pixel_to_cam(u, v, depth, fx, fy, cx, cy):
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    z = depth
    return np.array([x, y, z], dtype=np.float32)

def _frustum_corners_world(T_c2w: np.ndarray, intr: dict, near: float, far: float) -> np.ndarray:
    W = int(intr["W"]); H = int(intr["H"])
    fx = float(intr["fx"]); fy = float(intr["fy"]); cx = float(intr["cx"]); cy = float(intr["cy"])
    tl = (0.0, 0.0); tr = (W - 1.0, 0.0); br = (W - 1.0, H - 1.0); bl = (0.0, H - 1.0)
    n_tl = _pixel_to_cam(*tl, near, fx, fy, cx, cy)
    n_tr = _pixel_to_cam(*tr, near, fx, fy, cx, cy)
    n_br = _pixel_to_cam(*br, near, fx, fy, cx, cy)
    n_bl = _pixel_to_cam(*bl, near, fx, fy, cx, cy)
    f_tl = _pixel_to_cam(*tl, far,  fx, fy, cx, cy)
    f_tr = _pixel_to_cam(*tr, far,  fx, fy, cx, cy)
    f_br = _pixel_to_cam(*br, far,  fx, fy, cx, cy)
    f_bl = _pixel_to_cam(*bl, far,  fx, fy, cx, cy)
    C = np.stack([n_tl, n_tr, n_br, n_bl, f_tl, f_tr, f_br, f_bl], axis=0)  # (8,3)
    R = T_c2w[:3, :3]; t = T_c2w[:3, 3]
    return C @ R.T + t  # to world

def _align_matrix_z_to_vec(direction: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 transform that rotates +Z to 'direction' (unit) in world.
    """
    d = direction / (np.linalg.norm(direction) + 1e-9)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.allclose(d, z):
        R = np.eye(3, dtype=np.float32)
    elif np.allclose(d, -z):
        R = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # 180° around X
    else:
        v = np.cross(z, d)
        c = float(np.dot(z, d))
        s = float(np.linalg.norm(v))
        vx = np.array([
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],    0]
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + vx + vx @ vx * ((1.0 - c) / (s**2 + 1e-12))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    return T

def _edge_cylinders_from_corners(corners: np.ndarray,
                                 radius: float,
                                 sections: int,
                                 color_rgba: tuple) -> list:
    """
    Build slender cylinders along the 12 frustum edges.

    Args:
        corners: (8,3) world-space corner points in order [n_tl, n_tr, n_br, n_bl, f_tl, f_tr, f_br, f_bl]
        radius:  cylinder radius (meters)
        sections: # of radial segments (>= 6 looks fine)
        color_rgba: per-face RGBA

    Returns:
        list of trimesh.Trimesh cylinders already colored
    """
    n0,n1,n2,n3,f0,f1,f2,f3 = range(8)
    edges = [(n0,n1),(n1,n2),(n2,n3),(n3,n0),
             (f0,f1),(f1,f2),(f2,f3),(f3,f0),
             (n0,f0),(n1,f1),(n2,f2),(n3,f3)]
    out = []
    col = np.array(color_rgba, dtype=np.uint8)
    for a, b in edges:
        p0, p1 = corners[a], corners[b]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-8:
            continue
        mid = 0.5 * (p0 + p1)
        # unit cylinder along +Z, then scale/rotate/translate
        cyl = trimesh.creation.cylinder(radius=radius, height=1.0, sections=max(6, int(sections)))
        # color
        cyl.visual.face_colors = np.tile(col[None, :], (len(cyl.faces), 1))
        S = np.eye(4, dtype=np.float32); S[2, 2] = L
        R = _align_matrix_z_to_vec(d)
        Tt = np.eye(4, dtype=np.float32); Tt[:3, 3] = mid
        M = Tt @ R @ S
        cyl.apply_transform(M)
        out.append(cyl)
    return out


def _near_plane_textured_quad(
    corners: np.ndarray,
    image_path: Path,
    *,
    facing: str = "both",  # "out" | "in" | "both"
) -> trimesh.Trimesh:
    """
    Textured quad on the frustum's near plane.

    Args:
        corners: (8,3) frustum corners in world (n_tl,n_tr,n_br,n_bl,f_tl,f_tr,f_br,f_bl)
        image_path: path to the RGB file
        facing: "out" -> face outward, "in" -> face inward, "both" -> double-sided

    Returns:
        trimesh.Trimesh with texture visuals; if "both", reversed faces are appended.
    """
    from PIL import Image
    from trimesh.visual import texture as ttex

    # near corners in TL,TR,BR,BL order
    V = np.stack([corners[0], corners[1], corners[2], corners[3]], axis=0)  # (4,3)

    # UVs (flip V so the image looks upright)
    uv = np.array([[0.0, 1.0],
                   [1.0, 1.0],
                   [1.0, 0.0],
                   [0.0, 0.0]], dtype=np.float32)

    # Faces: choose winding to control normal direction
    F_out = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)  # points "out"
    F_in  = np.array([[2, 1, 0], [3, 2, 0]], dtype=np.int64)  # points "in"

    if facing == "in":
        F = F_in
        uv_all = uv
    elif facing == "both":
        F = np.vstack([F_out, F_in])
        uv_all = np.vstack([uv, uv])
    else:
        F = F_out
        uv_all = uv

    img = Image.open(str(image_path)).convert("RGB")

    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    material = ttex.SimpleMaterial(image=img)

    # Best-effort: ask exporter to mark the material as double-sided
    try:
        material.doubleSided = (facing == "both")
    except Exception:
        pass

    mesh.visual = ttex.TextureVisuals(uv=uv_all, image=img, material=material)
    return mesh



def write_glb_pointcloud_with_frusta(
    in_ply: Path,
    out_glb: Path,
    poses_meta: list,
    intr: dict,
    *,
    near: float = 0.06,
    far: float = 0.22,
    every: int = 1,                 # subsample frusta if many frames
    # ---- frustum style ----
    frustum_style: str = "solid",   # "solid" | "wire" | "none"
    frustum_color=(20, 20, 20, 150),
    frustum_edge_radius: float = 0.004,
    frustum_edge_sections: int = 10,
    # ---- textured near-plane ----
    image_plane: str = "none",      # "near" | "none"
    image_dir = None,
    image_name_fmt: str = "rgb_{:05d}.png",
    texture_every: int = 1,         # texture only some frusta
    max_textured = 120, # cap textures to avoid huge GLBs
    image_facing: str = "both",
    # ---- base cloud thinning ----
    point_subsample: int = 1,
    # ---- path & markers ----
    visit: np.ndarray = None,
    path_every: int = 1,
    path_radius: float = 0.015,
    path_color=(30, 110, 255, 255),
    anchors: np.ndarray = None,
    anchor_radius: float = 0.05,
    anchor_color=(255, 20, 20, 255),
    mark_first_last: bool = True,
    first_color=(30, 220, 30, 255),
    last_color=(255, 30, 255, 255),
    marker_radius: float = 0.07,
):
    """
    Export a single .glb scene including:
      • global point cloud (POINTS)
      • frusta as either:
          - 'solid'  : merged triangular meshes (old behavior)
          - 'wire'   : 12 thin cylinders per frustum (edges-only; no occlusion)
          - 'none'   : don't draw frusta
      • optional textured near-plane per (subsampled) frustum using rendered RGB
      • optional path cylinders, anchor spheres, first/last markers

    Notes
    -----
    • Wire mode avoids hiding the cloud. Tune `frustum_edge_radius` to taste.
    • Texturing every frustum can bloat the GLB. Use `texture_every`/`max_textured`.
    • Textured quads follow near-plane geometry and use `image_name_fmt` + `frame`.
    """
    in_ply = Path(in_ply); out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()

    # 1) Base point cloud
    V, C = _plyfile_load_xyz_rgb(in_ply)
    if point_subsample > 1 and V.shape[0] > 0:
        V = V[::point_subsample]
        C = C[::point_subsample]
    if V.shape[0] > 0:
        rgba = np.concatenate([np.clip(C, 0, 255), np.full((C.shape[0], 1), 255.0)], axis=1).astype(np.uint8)
        pc = trimesh.points.PointCloud(V, colors=rgba)
        scene.add_geometry(pc, node_name="global_point_cloud")

    step = max(1, int(every))
    col_rgba = np.array(frustum_color, dtype=np.uint8)
    if col_rgba.size == 3:
        col_rgba = np.concatenate([col_rgba, [255]]).astype(np.uint8)

    # 2) Frusta
    if frustum_style in ("solid", "wire"):
        if frustum_style == "solid":
            # merge all frusta triangles into one mesh
            V_all = []; F_all = []; off = 0
            F_template = np.array([
                [0,1,2],[0,2,3],   # near
                [4,5,6],[4,6,7],   # far
                [0,1,5],[0,5,4],
                [1,2,6],[1,6,5],
                [2,3,7],[2,7,6],
                [3,0,4],[3,4,7],
            ], dtype=np.int64)
            for m in poses_meta[::step]:
                T = np.asarray(m["T_c2w"], np.float32)
                corners = _frustum_corners_world(T, intr, near, far)  # (8,3)
                V_all.append(corners)
                F_all.append(F_template + off)
                off += 8
            if V_all:
                V_fr = np.vstack(V_all); F_fr = np.vstack(F_all)
                frusta = trimesh.Trimesh(vertices=V_fr, faces=F_fr, process=False)
                frusta.visual.face_colors = np.tile(col_rgba[None, :], (len(frusta.faces), 1))
                scene.add_geometry(frusta, node_name="camera_frusta_solid")
        else:
            # wire: add cylinders along edges per frustum
            for i, m in enumerate(poses_meta[::step]):
                T = np.asarray(m["T_c2w"], np.float32)
                corners = _frustum_corners_world(T, intr, near, far)
                cyls = _edge_cylinders_from_corners(
                    corners,
                    radius=float(frustum_edge_radius),
                    sections=int(frustum_edge_sections),
                    color_rgba=tuple(int(x) for x in col_rgba),
                )
                for j, g in enumerate(cyls):
                    scene.add_geometry(g, node_name=f"frustum_{i}_edge_{j}")

    # 3) Textured near planes (subsample to keep size reasonable)
    if image_plane == "near" and image_dir is not None:
        tex_step = max(1, int(texture_every))
        added = 0
        for m in poses_meta[::tex_step]:
            if max_textured is not None and added >= int(max_textured):
                break
            frame = int(m.get("frame", -1))
            if frame < 0:
                continue
            img_path = Path(image_dir) / image_name_fmt.format(frame)
            if not img_path.exists():
                continue
            T = np.asarray(m["T_c2w"], np.float32)
            corners = _frustum_corners_world(T, intr, near, far)
            quad = _near_plane_textured_quad(corners, img_path, facing=image_facing)
            scene.add_geometry(quad, node_name=f"frustum_near_tex_{frame}")
            added += 1

    # 4) Path as thin cylinders
    if visit is not None and np.asarray(visit).size >= 6 and path_radius > 0:
        Vv = np.asarray(visit, dtype=np.float32)
        idx = np.arange(0, Vv.shape[0], max(1, int(path_every)))
        if idx[-1] != Vv.shape[0]-1:
            idx = np.append(idx, Vv.shape[0]-1)
        base_cyl = trimesh.creation.cylinder(radius=path_radius, height=1.0, sections=12)
        pc_rgba = np.array(path_color, dtype=np.uint8)
        if pc_rgba.size == 3:
            pc_rgba = np.concatenate([pc_rgba, [255]]).astype(np.uint8)
        base_cyl.visual.face_colors = np.tile(pc_rgba[None, :], (len(base_cyl.faces), 1))
        for i in range(len(idx) - 1):
            p0, p1 = Vv[idx[i]], Vv[idx[i+1]]
            d = p1 - p0
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            mid = 0.5 * (p0 + p1)
            S = np.eye(4, dtype=np.float32); S[2, 2] = L
            R = _align_matrix_z_to_vec(d)
            Tt = np.eye(4, dtype=np.float32); Tt[:3, 3] = mid
            M = Tt @ R @ S
            scene.add_geometry(base_cyl, transform=M, node_name=f"path_seg_{i}")

    # 5) Anchor spheres
    if anchors is not None:
        A = np.asarray(anchors, dtype=np.float32)
        if A.size > 0 and anchor_radius > 0:
            sph = trimesh.creation.icosphere(subdivisions=2, radius=anchor_radius)
            col = np.array(anchor_color, dtype=np.uint8)
            if col.size == 3:
                col = np.concatenate([col, [255]]).astype(np.uint8)
            sph.visual.face_colors = np.tile(col[None, :], (len(sph.faces), 1))
            for i, p in enumerate(A):
                Tt = np.eye(4, dtype=np.float32); Tt[:3, 3] = p
                scene.add_geometry(sph, transform=Tt, node_name=f"anchor_{i}")

    # 6) First/last markers
    if mark_first_last and len(poses_meta) >= 1 and marker_radius > 0:
        P0 = np.array(poses_meta[0]["T_c2w"], dtype=np.float32)[:3, 3]
        P1 = np.array(poses_meta[-1]["T_c2w"], dtype=np.float32)[:3, 3]
        sph_first = trimesh.creation.icosphere(subdivisions=3, radius=marker_radius)
        col_f = np.array(first_color, dtype=np.uint8)
        if col_f.size == 3:
            col_f = np.concatenate([col_f, [255]]).astype(np.uint8)
        sph_first.visual.face_colors = np.tile(col_f[None, :], (len(sph_first.faces), 1))
        T0 = np.eye(4, dtype=np.float32); T0[:3, 3] = P0
        scene.add_geometry(sph_first, transform=T0, node_name="first_pose")

        if len(poses_meta) > 1:
            sph_last = trimesh.creation.icosphere(subdivisions=3, radius=marker_radius)
            col_l = np.array(last_color, dtype=np.uint8)
            if col_l.size == 3:
                col_l = np.concatenate([col_l, [255]]).astype(np.uint8)
            sph_last.visual.face_colors = np.tile(col_l[None, :], (len(sph_last.faces), 1))
            T1 = np.eye(4, dtype=np.float32); T1[:3, 3] = P1
            scene.add_geometry(sph_last, transform=T1, node_name="last_pose")

    # 7) Export GLB
    scene.export(str(out_glb))