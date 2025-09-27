from pathlib import Path
import numpy as np
import pycolmap

# ---- camera param helpers ---------------------------------------------------

def _intr_to_params(intr: dict, model: str) -> np.ndarray:
    """
    Convert your intr dict -> parameter vector for the given COLMAP model.
    intr keys: W, H, fx, fy, cx, cy
    """
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])

    if model == "PINHOLE":
        # [fx, fy, cx, cy]
        return np.array([fx, fy, cx, cy], dtype=np.float64)
    elif model == "SIMPLE_PINHOLE":
        # [f, cx, cy] with f = avg(fx, fy)
        f = 0.5 * (fx + fy)
        return np.array([f, cx, cy], dtype=np.float64)
    elif model == "SIMPLE_RADIAL":
        # [f, cx, cy, k]  (we put k=0 by default)
        f = 0.5 * (fx + fy)
        return np.array([f, cx, cy, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported camera model: {model}")


def _world_to_cam_from_Tc2w(T_c2w: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    COLMAP expects cam_from_world (R_cw, t_cw) such that:
        x_cam = R_cw * x_world + t_cw
    You currently store T_c2w (camera->world), so invert:
        R_cw = R_wc^T
        t_cw = -R_wc^T * t_wc
    """
    Twc = np.asarray(T_c2w, dtype=np.float64)
    Rwc = Twc[:3, :3]
    twc = Twc[:3, 3]
    Rcw = Rwc.T
    tcw = -Rcw @ twc
    return Rcw, tcw


# ---- main API ---------------------------------------------------------------

def export_colmap_reconstruction(
    colmap_dir: Path,
    intr: dict,
    poses_meta: list,
    image_name_fmt: str = "rgb_{:05d}.png",
    camera_model: str = "PINHOLE",
    shared_camera: bool = True,
) -> None:
    """
    Build a minimal pycolmap.Reconstruction with a single camera and N images.
    No points3D are added (not needed if you just want cameras/poses).

    Args:
      colmap_dir: output folder (will contain cameras.txt, images.txt, points3D.txt)
      intr: dict with W,H,fx,fy,cx,cy (same intrinsics used for rendering)
      poses_meta: list of dicts with fields:
          - "frame": int
          - "T_c2w": 4x4 list/array (camera->world)
      image_name_fmt: how your image files are named on disk
      camera_model: one of {"PINHOLE","SIMPLE_PINHOLE","SIMPLE_RADIAL"}
      shared_camera: if True, reuse the same Camera for all images
    """
    colmap_dir = Path(colmap_dir)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    W, H = int(intr["W"]), int(intr["H"])
    cam_params = _intr_to_params(intr, camera_model)

    recon = pycolmap.Reconstruction()

    # Create (shared) camera
    cam = pycolmap.Camera(
        model=camera_model,
        width=W,
        height=H,
        params=cam_params,
        camera_id=1
    )
    recon.add_camera(cam)
    print(f"Camera test : {cam}")

    # Add images with proper cam_from_world
    for k, meta in enumerate(poses_meta, start=1):  # 1-indexed ids
        fidx = int(meta["frame"])
        name = image_name_fmt.format(fidx)

        Rcw, tcw = _world_to_cam_from_Tc2w(meta["T_c2w"])
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(Rcw), tcw)

        img = pycolmap.Image(
            id=k,
            name=name,
            camera_id=cam.camera_id if shared_camera else k,
            cam_from_world=cam_from_world
        )
        # No points2D/3D here; still mark registered so COLMAP writes pose
        img.registered = True
        recon.add_image(img)

        # If not sharing camera, create per-image camera with same params
        if not shared_camera:
            ci = pycolmap.Camera(
                model=camera_model, width=W, height=H, params=cam_params, camera_id=k
            )
            recon.add_camera(ci)

    # No points3D needed for “poses-only” export.
    # COLMAP requires a points3D.txt file, but an empty one is fine.
    recon.write(colmap_dir)
