# Command-line Options

This tool uses three config groups:
- **RenderCfg**: rendering, camera basics, dataset selection
- **CoverageCfg**: path/anchor planning & heading policy
- **QualityCfg**: per-frame gating for SfM-friendly images

Profiles in `src/profiles.py` set smart defaults. Any CLI flag overrides the profile.

> Shown defaults vary by profile; use `--help` to see the active defaults after a profile is applied.

---

## RenderCfg

| Flag | Type | Meaning |
|---|---|---|
| `--seed` | int | Random seed for reproducibility. |
| `--out-path` | str | Output directory root. |
| `--H`, `--W` | int | Final output resolution (pixels). Rendering happens at `H*SSAA × W*SSAA`. |
| `--SSAA` | int | Supersampling factor (≥1). RGB downsampled by area; depth by min-pool. |
| `--hfov-deg` | float | Horizontal FOV for both RGB & Depth (keeps intrinsics consistent). |
| `--pitch-start` | float | Base pitch (deg). |
| `--round-pitch-deg` | floats | Per-round pitch overrides, e.g. `--round-pitch-deg -5 -10`. If fewer than rounds, last is reused. If empty, use `pitch-start`. |
| `--pitch-jitter` | float | Std-dev of pitch jitter (deg) for normal (non-coverage) shots. |
| `--height-start-cm` | float | Camera height (cm) for round 0. |
| `--height-jitter-cm` | float | Height jitter (cm) for normal shots. |
| `--roll-jitter` | float | Roll jitter (deg) for normal shots. |
| `--dataset-type` | str | `{replicaCAD_baked, replica, hm3d_v2}`. |
| `--dataset-path` | str | Root folder of the selected dataset. |
| `--scene-id` | str | Scene identifier (folder name or scene file, per dataset). |

---

## CoverageCfg

### Path & anchors

| Flag | Type | Meaning |
|---|---|---|
| `--traj-mode` | str | `{video, random}`. “video” follows a smoothed/resampled path. |
| `--n-anchors` | int | Target number of anchors along the tour. Auto-tuner may adjust. |
| `--traj-stride-m` | float | Resample spacing of the path (m). Controls frame spacing. |
| `--min-anchor-sep-m` | float | Minimum separation between anchors (m). Auto-tuner may update. |
| `--anchor-min-clearance-m` | float | Min obstacle clearance at anchors (m). Auto-tuner may relax. |

### Rounds (heights, yaws, pitches)

| Flag | Type | Meaning |
|---|---|---|
| `--two-rounds` | bool | If set, perform a second pass. |
| `--height2-cm` | float | Camera height (cm) for round 1. |
| `--round-yaw-offset-deg` | floats | Per-round yaw biases, e.g. `--round-yaw-offset-deg 0 30`. Applied before smoothing. |
| `--n-views` | int | Target number of accepted views (soft budget across rounds). |
| `--max-attempts` | int | Global attempt cap (safety). |
| `--max-rejects-per-visit` | int | Skip to next waypoint after too many rejects. |

> Round pitches are set in **RenderCfg** via `--round-pitch-deg`.

### Heading policy (per-frame)

| Flag | Type | Meaning |
|---|---|---|
| `--yaw-mode` | str | `{tangent, target, mixed}`. |
| `--yaw-mixed-alpha` | float | Blend weight for `mixed` mode. |
| `--yaw-limit-deg` | float | Pre-limit on yaw changes along the base schedule. |
| `--yaw-post-limit-deg` | float | Post-accept per-frame change limit. |
| `--yaw-smooth-alpha` | float | EMA smoothing between accepted frames. |

### Anchor-driven refinement (optional)

| Flag | Type | Meaning |
|---|---|---|
| `--anchor-yaw-enable` | bool | Enable anchor-aware heading refinement. |
| `--anchor-yaw-strength` | float | Influence of refined anchor deltas (0..1). |
| `--anchor-yaw-max-deg` | float | Clamp per-anchor delta before interpolation. |
| `--anchor-yaw-tries` | int | Internal attempts for refinement heuristics. |

**Forward-clearance clamp (navmesh-based)**

| Flag | Type | Meaning |
|---|---|---|
| `--anchor-yaw-min-fwd-clear-m` | float | Minimum forward clearance to accept a yaw direction (0 to disable). |
| `--anchor-yaw-scan-deg` | float | Yaw scan range (deg). |
| `--anchor-yaw-scan-steps` | int | Steps across the scan range. |

**Depth-openness scan (render-based)**

| Flag | Type | Meaning |
|---|---|---|
| `--anchor-viz-scan-deg` | float | Openness scan range (deg). Set **0** to disable (keep seed yaw). |
| `--anchor-viz-scan-steps` | int | Steps across the range. |
| `--anchor-viz-pitch-deg` | float | Pitch for the scan renders (deg). |
| `--anchor-viz-center-frac` | float | Center crop fraction considered for openness. |
| `--anchor-viz-arrow-len-m` | float | Debug overlay length (m). |
| `--save-anchor-images` | bool | Save per-anchor debug snapshots. |

### Deterministic coverage (playlist)

| Flag | Type | Meaning |
|---|---|---|
| `--deterministic-coverage` | bool | Enable a fixed per-anchor shot playlist. |
| `--cov-offsets-deg` | floats | Per-anchor yaw offsets (deg), e.g. `20 -20 160 -160`. |
| `--cov-at-same-pos` | bool | If false, advance to next waypoint after accepting a coverage shot. |
| `--cov-max-retries` | int | Small in-place yaw nudges if planned shot fails gate. |

> `coverage_rounds` (which rounds consume the playlist) is set by profiles (e.g., `(0,)` or `(0,1)`).

### Look-at targets (for `target`/`mixed`)

| Flag | Type | Meaning |
|---|---|---|
| `--min-geo`, `--max-geo` | float | Geodesic distance band for look-at sampling (m). |

---

## QualityCfg (gates)

All accepted frames must pass these gates; they’re tuned for SfM.

| Flag | Type | Meaning |
|---|---|---|
| `--min-valid-frac` | float | Fraction of finite depth required. |
| `--min-rel-depth-std` | float | Normalized depth variation (encourages 3D structure). |
| `--min-normal-disp` | float | Normal dispersion (rejects near-planar/degenerate views). |
| `--min-grad-mean` | float | Avg image gradient magnitude (reject blur/flat). |
| `--min-kpts` | int | Minimum number of detected features. |
| `--grid-nx`, `--grid-ny` | int | Grid for uniform keypoint coverage. |
| `--min-kpts-per-cell` | int | Per-cell minimum (0 disables per-cell gating). |

---

## Examples

**Two rounds, deterministic first, different pitches/heights** (Replica):

```bash
pixi run python -m src.main \
  --profile small_room_dense_2round \
  --dataset-type replica \
  --dataset-path habitat-data/replica \
  --scene-id room_2 \
  --out-path export-replica/room_2
```

**HM3D v2 minival multi-room**:

```bash
pixi run python -m src.main \
  --profile multi_rooms_dense_2round \
  --dataset-type hm3d_v2 \
  --dataset-path habitat-data/hm3d_v2_minival \
  --scene-id 00800-TEEsavR23oF \
  --out-path export-hm3d-v2-minival/00800
```

**Make coverage stronger on the first round only**:

```bash
pixi run python -m src.main \
  --profile small_room_dense_1round \
  --deterministic-coverage \
  --cov-offsets-deg 20 -20 160 -160
```

**Change per-round pitch from CLI**:

```bash
# round 0: -5°, round 1: -10°
--round-pitch-deg -5 -10
```
