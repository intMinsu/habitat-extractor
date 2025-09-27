#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# ---- Paths & profiles (match your examples) ----------------------------------
PY=python
MOD="src.main"

REPLICA_PATH="habitat-data/replica"
REPLICA_OUT_ROOT="export-replica"

HM3D_PATH="habitat-data/hm3d_v2_minival"
HM3D_OUT_ROOT="export-hm3d-v2-minival"

PROFILE_SMALL="small_room_dense_2round"
PROFILE_MULTI="multi_rooms_dense_2round"

# Optional extra CLI overrides (leave empty or add flags)
EXTRA_ARGS=${EXTRA_ARGS:-}

# Skip scenes that already finished (has poses_c2w.json)
SKIP_IF_DONE=${SKIP_IF_DONE:-1}

mkdir -p "logs" "$REPLICA_OUT_ROOT" "$HM3D_OUT_ROOT"

log() { echo -e "[$(date '+%H:%M:%S')] $*"; }

run_cmd() {
  local profile=$1 dataset_type=$2 dataset_path=$3 scene_id=$4 out_path=$5
  local tag="${dataset_type}_${profile}_${scene_id}"
  local logf="logs/${tag}.log"

  if [[ "$SKIP_IF_DONE" == "1" && -f "${out_path}/poses_c2w.json" ]]; then
    log "SKIP (done): $tag -> ${out_path}"
    return 0
  fi

  mkdir -p "$out_path"
  log "RUN  : $tag"
  log "OUT  : $out_path"
  set -x
  ${PY} -m ${MOD} \
    --profile "${profile}" \
    --dataset-type "${dataset_type}" \
    --dataset-path "${dataset_path}" \
    --scene-id "${scene_id}" \
    --out-path "${out_path}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${logf}"
  set +x
  log "DONE : $tag"
}

# ------------------------------------------------------------------------------
# (1) small_room_dense_2round on Replica: hotel_0, office_0..4, room_0..2
# ------------------------------------------------------------------------------
REPLICA_SMALL_SCENES=(
  hotel_0
  office_0 office_1 office_2 office_3 office_4
  room_0 room_1 room_2
)

for sid in "${REPLICA_SMALL_SCENES[@]}"; do
  run_cmd "${PROFILE_SMALL}" "replica" "${REPLICA_PATH}" "${sid}" "${REPLICA_OUT_ROOT}/${sid}"
done

# ------------------------------------------------------------------------------
# (2a) multi_rooms_dense_2round on Replica: apartment_0
# ------------------------------------------------------------------------------
REPLICA_LARGE_SCENES=(
  apartment_0 apartment_1 apartment_1
  frl_apartment_0
)
for sid in "${REPLICA_LARGE_SCENES[@]}"; do
  run_cmd "${PROFILE_MULTI}" "replica" "${REPLICA_PATH}" "${sid}" "${REPLICA_OUT_ROOT}/${sid}"
done

# ------------------------------------------------------------------------------
# (2b) multi_rooms_dense_2round on HM3D v2 minival: ALL scenes under the folder
# Scene IDs look like '00800-TEEsavR23oF'; out dir uses the numeric prefix (e.g., export-hm3d-v2-minival/00800)
# ------------------------------------------------------------------------------
shopt -s nullglob
for d in "${HM3D_PATH}"/*/; do
  sid="$(basename "$d")"                 # e.g., 00800-TEEsavR23oF
  short_id="${sid%%-*}"                  # e.g., 00800
  out_dir="${HM3D_OUT_ROOT}/${short_id}" # e.g., export-hm3d-v2-minival/00800
  run_cmd "${PROFILE_MULTI}" "hm3d_v2" "${HM3D_PATH}" "${sid}" "${out_dir}"
done
shopt -u nullglob

log "All batches finished."
