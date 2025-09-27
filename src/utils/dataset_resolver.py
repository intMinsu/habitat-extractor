import os, re
from pathlib import Path

def map_replica_cad_baked_scene_id(token: str) -> str:
    """'sc0_00' -> 'Baked_sc0_staging_00.scene_instance.json' (pass-through for full path)."""
    if token.endswith(".scene_instance.json"):
        return token
    m = re.fullmatch(r"sc(\d+)[\-_]?(\d+)", token.strip())
    if not m:
        return token
    sc = int(m.group(1)); idx = int(m.group(2))
    return f"Baked_sc{sc}_staging_{idx:02d}.scene_instance.json"

def map_hm3d_v2_scene_id(root: Path, token: str) -> str:
    """
    HM3D v2 mapping:
      - '00800-TEEsavR23oF' -> <root>/hm3d-minival-habitat-v2/00800-TEEsavR23oF/TEEsavR23oF.basis.glb
      - full '*.glb' path -> pass-through
    """
    if token.endswith(".glb"):
        return token
    folder = root / "hm3d-minival-habitat-v2" / token
    rid = token.split("-")[-1]  # 'TEEsavR23oF'
    glb = folder / f"{rid}.basis.glb"
    return str(glb)

def resolve_dataset_and_scene(args):
    root = Path(args.dataset_path)

    if args.dataset_type == "replicaCAD_baked":
        dataset_cfg = root / "replicaCAD_baked.scene_dataset_config.json"
        if not dataset_cfg.exists():
            raise FileNotFoundError(f"Missing dataset config: {dataset_cfg}")
        scene_id = map_replica_cad_baked_scene_id(args.scene_id)
        return str(dataset_cfg), scene_id

    elif args.dataset_type == "replica":
        dataset_cfg = Path(args.dataset_path) / "replica.scene_dataset_config.json"
        if not dataset_cfg.exists():
            raise FileNotFoundError(f"Missing Replica dataset config: {dataset_cfg}")
        scene_id = args.scene_id.strip()
        return str(dataset_cfg), scene_id

    elif args.dataset_type == "hm3d_v2":
        cfg_minival = root / "hm3d-minival-semantic-configs-v2" / "hm3d_annotated_minival_basis.scene_dataset_config.json"
        cfg_full    = root / "hm3d-minival-semantic-configs-v2" / "hm3d_annotated_basis.scene_dataset_config.json"
        dataset_cfg = cfg_minival if cfg_minival.exists() else cfg_full
        if not dataset_cfg.exists():
            raise FileNotFoundError(f"Missing HM3D dataset config in {dataset_cfg.parent}")
        scene_glb = map_hm3d_v2_scene_id(root, args.scene_id)
        if not Path(scene_glb).exists():
            raise FileNotFoundError(f"HM3D scene glb not found: {scene_glb}")
        return str(dataset_cfg), scene_glb

    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")