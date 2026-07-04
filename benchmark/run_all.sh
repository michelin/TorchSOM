#!/usr/bin/env bash
# Full SOM benchmark sweep: 4 backends × {cpu,cuda} × 9 dataset sizes (rectangular 25x15).
#
# SUPERSEDED for the TE re-run: run_rectangular.sh now covers rectangular 25x15 (all 9
# datasets) AND 90x70, so use run_rectangular.sh + run_hexagonal.sh instead. Running this
# too would duplicate rectangular-25x15 under a differently-named sweep_<STAMP>/ folder.
# Kept as the original single-size full sweep; somoclu is now GPU-only (cuda runs only).
#
#   datasets : [300, 5000, 20000] samples × [4, 50, 300] features  (blobs_<n>_<f>)
#   backends : torchsom, minisom, minisom_jit, somoclu (somoclu is GPU-only, cuda runs only)
#
# CPU runs torchsom + minisom + minisom_jit; CUDA runs torchsom + somoclu (somoclu is the
# GPU-only baseline; MiniSom / MiniSom-JIT are CPU-only, so running them under cuda would
# just re-do the CPU work). The base config configs/benchmark.yaml is used read-only as a
# template: each run gets a generated config with data_name / device / use_* overridden.
#
# Logs every run; continues on failure (e.g. a missing somoclu binary). Outputs are
# grouped per sweep under results/sweep_<STAMP>/ and logs/sweep_<STAMP>/.
set -uo pipefail
cd "$(dirname "$0")"                       # runner resolves data at ../data/benchmark/*.csv

BASE_CONFIG="configs/benchmark.yaml"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS_ROOT="results/sweep_${STAMP}"
LOG_DIR="logs/sweep_${STAMP}"; CFG_DIR="${LOG_DIR}/configs"
mkdir -p "$CFG_DIR"; SUMMARY="${LOG_DIR}/summary.log"

# Cheap -> expensive (samples × features) so early feedback comes fast and the
# multi-hour MiniSom configs run last.
DATASETS=( 300_4 300_50 5000_4 20000_4 300_300 5000_50 20000_50 5000_300 20000_300 )

# Skip the cuda half entirely if no GPU is visible (the runner would otherwise silently
# fall back to CPU and pollute the "cuda" results).
CUDA_OK=0
if uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CUDA_OK=1
else
  echo "WARNING: CUDA unavailable — cuda runs skipped." | tee -a "$SUMMARY"
fi

make_config() {   # data_name device use_minisom use_minisom_jit use_somoclu outfile
  # Override only these 6 lines; everything else (random_seed, n_iter, the whole som:
  # section, ...) is inherited from BASE_CONFIG. The trailing ':' anchors keep the
  # use_minisom: rule from matching use_minisom_jit:.
  sed -E \
    -e "s|^([[:space:]]*data_name:).*|\1 $1|" \
    -e "s|^([[:space:]]*device:).*|\1 $2|" \
    -e "s|^([[:space:]]*use_torchsom:).*|\1 true|" \
    -e "s|^([[:space:]]*use_minisom:).*|\1 $3|" \
    -e "s|^([[:space:]]*use_minisom_jit:).*|\1 $4|" \
    -e "s|^([[:space:]]*use_somoclu:).*|\1 $5|" \
    "$BASE_CONFIG" > "$6"
}

echo "Sweep $STAMP — results: $RESULTS_ROOT | logs: $LOG_DIR" | tee -a "$SUMMARY"
for ds in "${DATASETS[@]}"; do
  for dev in cpu cuda; do
    if [[ "$dev" == cuda && "$CUDA_OK" -ne 1 ]]; then
      echo "SKIP  blobs_${ds} cuda (no GPU)" | tee -a "$SUMMARY"; continue; fi
    # CPU: torchsom + minisom + minisom_jit (somoclu is GPU-only). CUDA: torchsom + somoclu.
    if [[ "$dev" == cpu ]]; then mini=true; jit=true; som=false; else mini=false; jit=false; som=true; fi
    cfg="${CFG_DIR}/blobs_${ds}__${dev}.yaml"; log="${LOG_DIR}/blobs_${ds}__${dev}.log"
    make_config "blobs_${ds}" "$dev" "$mini" "$jit" "$som" "$cfg"
    echo "RUN   blobs_${ds} ${dev} -> $log" | tee -a "$SUMMARY"
    start=$(date +%s)
    uv run python benchmark.py --config-path "$cfg" --output-dir "$RESULTS_ROOT" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}; dur=$(( $(date +%s) - start ))   # python's exit code, not tee's
    if [[ $rc -eq 0 ]]; then echo "PASS  blobs_${ds} ${dev} (${dur}s)" | tee -a "$SUMMARY"
    else echo "FAIL  blobs_${ds} ${dev} (rc=$rc, ${dur}s) — see $log" | tee -a "$SUMMARY"; fi
  done
done
echo "Done. Summary: $SUMMARY" | tee -a "$SUMMARY"
