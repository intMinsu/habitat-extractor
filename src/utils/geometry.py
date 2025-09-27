import numpy as np
import math

def unproject_depth_to_points_cam(depth, K):
    Hh, Ww = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(Ww, dtype=np.float32),
                         np.arange(Hh, dtype=np.float32))
    Z = depth
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    return np.stack([X, Y, Z], -1).reshape(-1, 3)

def normals_from_depth(depth, K):
    Hh, Ww = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(Ww, dtype=np.float32),
                         np.arange(Hh, dtype=np.float32))
    Z = depth
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    dXdu, dYdu, dZdu = np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)
    dXdv, dYdv, dZdv = np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)
    t_u = np.stack([dXdu, dYdu, dZdu], -1)
    t_v = np.stack([dXdv, dYdv, dZdv], -1)
    n = np.cross(t_u, t_v)
    n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)
    return n

def intrinsics_from_spec(rgb_spec):
    Hs = int(rgb_spec.resolution[0])
    Ws = int(rgb_spec.resolution[1])
    hfov_rad = float(rgb_spec.hfov) * math.pi / 180.0
    vfov_rad = 2.0 * math.atan((Hs / Ws) * math.tan(hfov_rad / 2.0))

    fx = (Ws / 2.0) / math.tan(hfov_rad / 2.0)
    fy = (Hs / 2.0) / math.tan(vfov_rad / 2.0)
    cx, cy = (Ws - 1) / 2.0, (Hs - 1) / 2.0

    K = [
        [float(fx), 0.0,       float(cx)],
        [0.0,       float(fy), float(cy)],
        [0.0,       0.0,       1.0      ],
    ]
    return {
        "W": int(Ws),
        "H": int(Hs),
        "hfov_deg": float(rgb_spec.hfov),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "K": K,
    }


def intrinsics_from_sensor(sim, rgb_uuid="rgba", hfov_deg_hint=None, fallback_resolution=None):
    """
    Return {W,H,fx,fy,cx,cy,K,hfov_deg} using whatever the current Habitat build exposes.
    Priority order for HFOV: sensor.fov -> sensor.hfov -> sensor.specification().hfov -> hfov_deg_hint -> 90
    Resolution comes from a real render, with fallbacks.
    """
    # 1) Get the sensor
    cam = sim._sensors[rgb_uuid]

    # 2) Resolution (prefer actual render to avoid stale specs)
    H = W = None
    try:
        obs = sim.get_sensor_observations()
        arr = obs[rgb_uuid]
        H, W = int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        # fallback to spec if available
        if hasattr(cam, "specification"):
            try:
                spec = cam.specification()
                H = int(spec.resolution[0])
                W = int(spec.resolution[1])
            except Exception:
                pass
        if (H is None or W is None) and fallback_resolution is not None:
            H, W = map(int, fallback_resolution)
    if H is None or W is None:
        raise RuntimeError("intrinsics_from_sensor: couldn't determine sensor resolution")

    # 3) HFOV (degrees)
    hfov_deg = None
    for attr in ("fov", "hfov"):
        if hasattr(cam, attr):
            try:
                hfov_deg = float(getattr(cam, attr))
                break
            except Exception:
                pass
    if hfov_deg is None and hasattr(cam, "specification"):
        try:
            hfov_deg = float(cam.specification().hfov)
        except Exception:
            pass
    if hfov_deg is None and hfov_deg_hint is not None:
        hfov_deg = float(hfov_deg_hint)
    if hfov_deg is None:
        hfov_deg = 90.0  # safe default

    # 4) Pinhole intrinsics from HFOV
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan((H / W) * math.tan(hfov * 0.5))
    fx = (W * 0.5) / math.tan(hfov * 0.5)
    fy = (H * 0.5) / math.tan(vfov * 0.5)
    # cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    cx, cy = W / 2.0, H / 2.0

    K = [
        [float(fx), 0.0,       float(cx)],
        [0.0,       float(fy), float(cy)],
        [0.0,       0.0,       1.0      ],
    ]
    return {
        "W": int(W), "H": int(H), "hfov_deg": float(hfov_deg),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "K": K,
    }