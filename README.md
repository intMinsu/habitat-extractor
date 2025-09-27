# habitat-extractor

Export SFM-friendly RGB/Depth/Normal/Pointclouds and COLMAP models from Habitat scenes.

## Quick start

```bash
python -m src.main \
  --profile standard \
  --n-views 20 \
  --dataset-type replica \
  --dataset-path habitat-data/replica \
  --scene-id room_0 \
  --out-path export-replica/room_0_testvis