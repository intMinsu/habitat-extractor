# Habitat Scene Extractor

Export SfM-friendly views from Habitat scenes (Replica, HM3D v2, ReplicaCAD baked).  
The pipeline builds a smooth “video-like” trajectory, auto-tunes and spreads anchors, plans per-frame headings, gates low-quality views, writes COLMAP, and ships nice visualizations (PLY/GLB, bird’s-eye).

## Quickstart (with `pixi`)

This repo includes a `pixi.toml`. If you don’t have pixi:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
# restart your shell if needed
```

Create the env and run one scene:

```bash
# 1) install the environment from pixi.toml
pixi install

# 2) run an example (Replica / room_0) using the default pixi env
pixi run python -m src.main \
  --profile small_room_dense_2round \
  --dataset-type replica \
  --dataset-path habitat-data/replica \
  --scene-id room_0 \
  --out-path export-replica/room_0
```

> Tip: No pixi? A standard Python 3.8+ venv works too—just install the same deps listed in `pixi.toml`.

Outputs go to `--out-path`:
- `images/`, `depth/`, `normal/`, `mask/`
- per-view point clouds `points_exr/` & merged `global_points_ply/`
- COLMAP folder `sparse/`
- trajectory viz `traj_vis/` (PLY overlay, GLB with frusta, bird’s-eye)

## Profiles (presets)

Profiles are sensible bundles of defaults; any CLI flag still overrides them. See `src/profiles.py` for exact values.

- **standard**  
  Balanced defaults; deterministic coverage enabled by default.

- **small_room_dense_1round**  
  Dense capture for compact single rooms, one pass. Uses anchor-aware yaws and deterministic coverage (offset-major). Good for small/occluded interiors.

- **small_room_dense_2round**  
  Two passes at different heights/pitches, deterministic coverage first, then normal sampling. Complements viewpoints in tight spaces.

- **multi_rooms_dense_1round**  
  (For multi-room apartments/offices) One dense pass; more anchors and slightly longer stride for coverage across several rooms.

- **multi_rooms_dense_2round**  
  Two passes across multi-room scenes; round-specific yaw offsets and pitches to diversify parallax and height.

- **fast_debug**  
  Low-res, few views; great for sanity checks and quick iteration.

For detailed CLI options (and how profiles map to config fields), see **[docs/options.md](docs/options.md)**.

## Batch all scenes (`run_all_scenes.sh`)

We provide a convenience script to process multiple scenes with the right profiles. It:
- uses the same dataset roots shown below,
- writes logs to `./logs/`,
- skips scenes that already have `poses_c2w.json` (toggle via `SKIP_IF_DONE`),
- lets you pass `EXTRA_ARGS="..."` to append flags to every run.

Run it:

```bash
chmod +x run_all_scenes.sh
./run_all_scenes.sh

# Optional:
EXTRA_ARGS="--n-views 220 --SSAA 2" ./run_all_scenes.sh
SKIP_IF_DONE=0 ./run_all_scenes.sh
```

## Datasets

Where to get them and how to lay out folders: see **[docs/dataset.md](docs/dataset.md)**.
