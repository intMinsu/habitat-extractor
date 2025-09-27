import numpy as np
import quaternion as nq
import magnum as mn

def _quat_xyzw_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Convert a quaternion given in **(x, y, z, w)** order to a 3×3 rotation matrix.

    Robust to zero-norm (returns I).
    """
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - s * (yy + zz),     s * (xy - wz),         s * (xz + wy)],
        [    s * (xy + wz),   1.0 - s * (xx + zz),       s * (yz - wx)],
        [    s * (xz - wy),       s * (yz + wx),     1.0 - s * (xx + yy)],
    ], dtype=np.float32)


def rotmat_from_quat(q) -> np.ndarray:
    """
    Return a 3×3 rotation matrix from either a **Magnum quaternion** or a **numpy-quaternion**.
    Also supports generic objects exposing (x,y,z,w) or a 4-vector in (x,y,z,w).

    Priority:
      1) Magnum: `q.to_matrix()` if available (fast, exact).
      2) numpy-quaternion: `nq.as_rotation_matrix(q)`.
      3) Magnum fallback: synthesize R by rotating basis vectors with `transform_vector`.
      4) Generic: pull (x,y,z,w) fields or a length-4 array/list and call `_quat_xyzw_to_R`.
    """
    # 1) Magnum fast path
    if hasattr(q, "to_matrix"):
        return np.array(q.to_matrix(), dtype=np.float32)

    # 2) numpy-quaternion
    if (nq is not None) and (type(q).__name__ == "quaternion"):
        return np.array(nq.as_rotation_matrix(q), dtype=np.float32)

    # 3) Magnum fallback via rotating basis vectors
    if hasattr(q, "transform_vector"):
        e0, e1, e2 = mn.Vector3(1.0, 0.0, 0.0), mn.Vector3(0.0, 1.0, 0.0), mn.Vector3(0.0, 0.0, 1.0)
        v0 = q.transform_vector(e0)
        v1 = q.transform_vector(e1)
        v2 = q.transform_vector(e2)
        return np.array([[v0.x, v1.x, v2.x],
                         [v0.y, v1.y, v2.y],
                         [v0.z, v1.z, v2.z]], dtype=np.float32)

    # 4) Generic fields / array in (x,y,z,w)
    if hasattr(q, "vector") and hasattr(q, "scalar"):
        x, y, z = float(q.vector[0]), float(q.vector[1]), float(q.vector[2])
        w = float(q.scalar)
        return _quat_xyzw_to_R(x, y, z, w)
    if all(hasattr(q, a) for a in ("x", "y", "z", "w")):
        return _quat_xyzw_to_R(float(q.x), float(q.y), float(q.z), float(q.w))

    arr = np.array(q, dtype=np.float64).ravel()
    if arr.size == 4:
        x, y, z, w = map(float, arr)
        return _quat_xyzw_to_R(x, y, z, w)

    # Fallback: identity
    return np.eye(3, dtype=np.float32)

def rotate_vec3_from_quat_axis(v3, q) -> np.ndarray:
    """
    Rotate a 3D vector `v3` by quaternion `q` (Magnum or numpy-quaternion).
    Returns a (3,) float32 numpy array.
    """
    v3 = np.asarray(v3, dtype=np.float64).reshape(-1)
    if v3.size != 3:
        raise ValueError(f"rotate_vec3_from_quat_axis expected 3 values, got {v3.size}")
    v3.astype(np.float32)

    # Magnum fast path
    if hasattr(q, "transform_vector"):
        w = q.transform_vector(mn.Vector3(float(v3[0]), float(v3[1]), float(v3[2])))
        return np.array([w.x, w.y, w.z], dtype=np.float32)
    # Generic: via rotation matrix
    R = rotmat_from_quat(q)
    return (R @ v3).astype(np.float32)

def T_c2w_from_sensor_state(s):
    """
    Build the **camera→world** transform in **OpenCV camera convention** (+X right, +Y down, +Z forward).

    Habitat/Magnum camera convention is (+X right, +Y up, −Z forward).
    To convert to OpenCV’s (+X right, +Y down, +Z forward), we right-multiply the
    Habitat rotation `R_wc_hab` by the fixed basis flip:

          F = diag(1, −1, −1)

    so `R_wc_cv = R_wc_hab @ F`. Translation is unchanged.

    Returns:
        T (4×4 float32): camera(OpenCV)→world transform.
    """
    # translation (Magnum Vector3 or array-like)
    try:
        t = np.array([float(s.position.x), float(s.position.y), float(s.position.z)], dtype=np.float32)
    except AttributeError:
        t = np.array(s.position, dtype=np.float32).reshape(3)

    # rotation in Habitat convention
    R_wc_hab = rotmat_from_quat(s.rotation)

    # flip basis to OpenCV convention
    F = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_wc_cv = R_wc_hab @ F

    # assemble homogeneous transform
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_wc_cv
    T[:3,  3] = t
    return T
