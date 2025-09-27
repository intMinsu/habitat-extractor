# Datasets

This project supports **Replica**, **HM3D v2**, and **ReplicaCAD (baked lighting)**.

## Where to download

- Replica: <https://github.com/facebookresearch/Replica-Dataset>  
- HM3D / Habitat-MP3D research repo: <https://github.com/matterport/habitat-matterport-3dresearch>  
- ReplicaCAD (baked lighting): <https://aihabitat.org/datasets/replica_cad/>

> Follow each dataset’s license & access instructions.

## Folder layout (example)

```
habitat-extractor/habitat-data
├── hm3d_v2_minival
│   ├── hm3d-minival-glb-v2
│   ├── hm3d-minival-habitat-v2
│   ├── hm3d-minival-semantic-annots-v2
│   └── hm3d-minival-semantic-configs-v2
├── replica
│   ├── apartment_0
│   ├── apartment_1
│   ├── apartment_2
│   ├── frl_apartment_0
│   ├── frl_apartment_1
│   ├── frl_apartment_2
│   ├── frl_apartment_3
│   ├── frl_apartment_4
│   ├── frl_apartment_5
│   ├── hotel_0
│   ├── office_0
│   ├── office_1
│   ├── office_2
│   ├── office_3
│   ├── office_4
│   ├── replica.scene_dataset_config.json
│   ├── room_0
│   ├── room_1
│   └── room_2
└── replica_cad_baked_lighting
    ├── configs
    ├── LICENSE.txt
    ├── navmeshes
    ├── navmeshes_default
    ├── README.md
    ├── replicaCAD_baked.scene_dataset_config.json
    ├── stages
    ├── stages_uncompressed
    ├── urdf
    └── urdf_uncompressed
```

## How to point the runner

- **Replica**
  - `--dataset-type replica`
  - `--dataset-path habitat-data/replica`
  - `--scene-id <folder name>` (e.g., `room_0`, `office_4`, `hotel_0`)

- **HM3D v2 (minival)**
  - `--dataset-type hm3d_v2`
  - `--dataset-path habitat-data/hm3d_v2_minival`
  - `--scene-id <folder name>` (e.g., `00800-TEEsavR23oF`)
  - In our batch script, outputs are grouped by numeric prefix, e.g. `export-hm3d-v2-minival/00800`.

- **ReplicaCAD (baked)**
  - `--dataset-type replicaCAD_baked`
  - `--dataset-path habitat-data/replica_cad_baked_lighting`
  - `--scene-id` is either a short ID (e.g., `sc0_00`) or a full `*.scene_instance.json` path.

> Navmeshes: HM3D v2 and Replica typically ship with navmeshes; ReplicaCAD baked also provides navmesh folders. **The auto-tuning and anchor placement logic rely on a connected navmesh**—if you see very few anchors, verify connectivity.
