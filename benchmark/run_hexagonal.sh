#!/usr/bin/env bash
# Hexagonal-topology benchmark sweep over 25x15 (all 9 datasets) and 90x70 (3 widest).
#
#   topology : hexagonal  (honored by ALL backends -- needs the benchmark.py topology
#              pass-through: MiniSom/JIT topology=, somoclu gridtype=, hexagonal somoclu TE)
#   sizes    : 25x15 (run first: cheap full-pipeline validation), then 90x70
#   datasets : 25x15 -> all 9 (blobs_<n>_<f>); 90x70 -> the 3 widest (300-feature) only
#   backends : torchsom, minisom, minisom_jit, somoclu (somoclu is GPU-only, cuda runs only)
#   devices  : cpu (torchsom + minisom + minisom_jit) + cuda (torchsom + somoclu)
#
# Companion to run_rectangular.sh; both share an identical body and differ only in the
# CONFIG block below. Each size gets its own sweep folder suffixed with <topology>_<size>,
# since the per-run results path encodes <topology> but NOT the map size.
#
# The base config configs/benchmark.yaml is the read-only template; each run gets a
# generated config with data_name / device / use_* / topology / x_size / y_size overridden.
# Logs every run; continues on failure (e.g. a missing somoclu binary).
#
# WARNING: online MiniSom scales ~linearly with neuron count; 90x70 has ~17x the neurons of
# 25x15. The largest cells (blobs_20000_300 @ 90x70 CPU) can take ~2-3 days each -- the full
# sweep is potentially a week-plus of wall-clock.
set -uo pipefail
cd "$(dirname "$0")"                       # runner resolves data at ../data/benchmark/*.csv

# ===== CONFIG (the only block that differs from run_rectangular.sh) ===================
TOPOLOGY="hexagonal"
# "size_label x_size y_size"  (cheap -> expensive map first)
SIZES=(
  "25x15 25 15"
  "90x70 90 70"
)
# Datasets depend on map size: the small map runs all 9 (samples x features), cheap ->
# expensive; the large map runs only the 3 widest (300-feature) datasets per paper scope.
DATASETS_SMALL=( 300_4 300_50 300_300 5000_4 5000_50 5000_300 20000_4 20000_50 20000_300 )
DATASETS_LARGE=( 300_300 5000_300 20000_300 )
# =====================================================================================

BASE_CONFIG="configs/benchmark.yaml"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

# Skip the cuda half entirely if no GPU is visible (the runner would otherwise silently
# fall back to CPU and pollute the "cuda" results).
CUDA_OK=0
if uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CUDA_OK=1
else
  echo "WARNING: CUDA unavailable -- cuda runs skipped (somoclu is GPU-only, so it is skipped too)."
fi

make_config() {   # data_name device use_minisom use_minisom_jit use_somoclu topology x_size y_size outfile
  # Override only these 9 lines; everything else (random_seed, n_iter, sigma, epochs, ...)
  # is inherited from BASE_CONFIG. The trailing ':' anchors keep the use_minisom: rule from
  # matching use_minisom_jit:.
  sed -E \
    -e "s|^([[:space:]]*data_name:).*|\1 $1|" \
    -e "s|^([[:space:]]*device:).*|\1 $2|" \
    -e "s|^([[:space:]]*use_torchsom:).*|\1 true|" \
    -e "s|^([[:space:]]*use_minisom:).*|\1 $3|" \
    -e "s|^([[:space:]]*use_minisom_jit:).*|\1 $4|" \
    -e "s|^([[:space:]]*use_somoclu:).*|\1 $5|" \
    -e "s|^([[:space:]]*topology:).*|\1 $6|" \
    -e "s|^([[:space:]]*x_size:).*|\1 $7|" \
    -e "s|^([[:space:]]*y_size:).*|\1 $8|" \
    "$BASE_CONFIG" > "$9"
}

for size in "${SIZES[@]}"; do
  read -r szlabel xs ys <<< "$size"
  # Small map sweeps all 9 datasets; the large (90x70) map only the 3 widest.
  if [[ "$szlabel" == "25x15" ]]; then ds_list=( "${DATASETS_SMALL[@]}" ); else ds_list=( "${DATASETS_LARGE[@]}" ); fi
  TAG="${TOPOLOGY}_${szlabel}"
  RESULTS_ROOT="results/sweep_${STAMP}_${TAG}"
  LOG_DIR="logs/sweep_${STAMP}_${TAG}"; CFG_DIR="${LOG_DIR}/configs"
  mkdir -p "$CFG_DIR"; SUMMARY="${LOG_DIR}/summary.log"
  echo "Sweep $TAG (${xs}x${ys}) -- results: $RESULTS_ROOT | logs: $LOG_DIR" | tee -a "$SUMMARY"
  for ds in "${ds_list[@]}"; do
    for dev in cpu cuda; do
      if [[ "$dev" == cuda && "$CUDA_OK" -ne 1 ]]; then
        echo "SKIP  blobs_${ds} ${TAG} cuda (no GPU)" | tee -a "$SUMMARY"; continue; fi
      # CPU: torchsom + minisom + minisom_jit (somoclu is GPU-only). CUDA: torchsom + somoclu
      # (minisom/jit are CPU-only).
      if [[ "$dev" == cpu ]]; then mini=true; jit=true; som=false; else mini=false; jit=false; som=true; fi
      cfg="${CFG_DIR}/blobs_${ds}__${dev}.yaml"; log="${LOG_DIR}/blobs_${ds}__${dev}.log"
      make_config "blobs_${ds}" "$dev" "$mini" "$jit" "$som" "$TOPOLOGY" "$xs" "$ys" "$cfg"
      echo "RUN   blobs_${ds} ${TAG} ${dev} -> $log" | tee -a "$SUMMARY"
      start=$(date +%s)
      uv run python benchmark.py --config-path "$cfg" --output-dir "$RESULTS_ROOT" 2>&1 | tee "$log"
      rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))   # python's exit code, not tee's
      if [[ $rc -eq 0 ]]; then echo "PASS  blobs_${ds} ${TAG} ${dev} (${dur}s)" | tee -a "$SUMMARY"
      else echo "FAIL  blobs_${ds} ${TAG} ${dev} (rc=$rc, ${dur}s) -- see $log" | tee -a "$SUMMARY"; fi
    done
  done
  echo "Sweep $TAG done. Summary: $SUMMARY" | tee -a "$SUMMARY"
done
echo "All ${TOPOLOGY} sweeps done. Logs under logs/sweep_${STAMP}_${TOPOLOGY}_*/"
