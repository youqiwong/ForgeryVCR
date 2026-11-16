#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"
# ============================================================
# run_expert_test.sh — merge a single-tool expert, build its full test QA
# (fake = casia2_tp, real = casia2_au), and run vLLM inference.
#
#   bash run_expert_test.sh <tool> <lora_checkpoint>
#       tool = notool | ela | fft | npp | zoom_in
#
# Output: $RESULTS/<tool>.jsonl  (feeds compute_gain_csv.py --tool NAME=...)
# Override DATA / OUT / TOOLMAPS via env if your layout differs.
# ============================================================
export RCCL_DEBUG=INFO
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_VERBOSITY=error
export MIOPEN_LOG_LEVEL=1
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1

TOOL="${1:?usage: bash run_expert_test.sh <notool|ela|fft|npp|zoom_in> <lora_checkpoint>}"
CKPT="${2:?missing <lora_checkpoint>}"

DATA="${DATA:-datasets/train/832x}"
OUT="${OUT:-workdir}"
RESULTS="$OUT/results"; mkdir -p "$RESULTS"
TOOLMAPS="${TOOLMAPS:-datasets/train/tool_maps}"

case "$TOOL" in
  notool)  TMDIR="" ;;
  ela)     TMDIR="$TOOLMAPS" ;;
  fft)     TMDIR="$TOOLMAPS" ;;
  npp)     TMDIR="$TOOLMAPS" ;;
  zoom_in) TMDIR="$OUT/zoomin_images" ;;
  *) echo "unknown tool: $TOOL"; exit 1 ;;
esac

# 1) merge LoRA -> <ckpt>-merged (skip if already merged)
MERGED="${CKPT}-merged"
if [ ! -d "$MERGED" ]; then
  echo ">>> merging LoRA: $CKPT"
  swift export --adapters "$CKPT" --merge_lora true
fi

# 2) in-domain test == the exact training QA (qa_<tool>.json, balanced real/fake).
#    Reuse it if present; otherwise rebuild it with --balance so it matches training.
QA="$RESULTS/qa_${TOOL}.json"
if [ ! -f "$QA" ]; then
  echo ">>> building balanced (in-domain) QA: $QA"
  ARGS="--tool $TOOL --fake_json $DATA/casia2_tp/casia2_tp.json --au_dir $DATA/casia2_au --out $QA --balance"
  [ -n "$TMDIR" ] && ARGS="$ARGS --toolmap_dir $TMDIR"
  python "$REPO_ROOT/experts/build_qa_dataset.py" $ARGS
else
  echo ">>> using existing training QA (in-domain): $QA"
fi

# 3) vLLM inference in resumable CHUNKS -> $RESULTS/<tool>.jsonl
#    Each chunk is a fresh `swift infer` process (frees host RAM between chunks),
#    and samples already present in the jsonl are skipped, so a crash just means
#    "re-run to continue". Tune via env: CHUNK / NPROC / MAX_NUM_SEQS.
RESULT="$RESULTS/${TOOL}.jsonl"
TODO="$RESULTS/.${TOOL}_todo.json"
CHUNK="${CHUNK:-3000}"
NPROC="${NPROC:-8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
echo ">>> resumable inference with $MERGED -> $RESULT (chunk=$CHUNK, nproc=$NPROC)"

set +e   # tolerate a chunk crash; the loop resumes from what's on disk
while true; do
  python "$REPO_ROOT/experts/filter_done.py" \
      --qa "$QA" --done "$RESULT" --out "$TODO" --limit "$CHUNK"
  n=$(python -c "import json;print(len(json.load(open('$TODO'))))")
  [ "$n" -eq 0 ] && { echo ">>> all samples done"; break; }

  before=$(wc -l < "$RESULT" 2>/dev/null || echo 0)
  TMP="$RESULT.part"; rm -f "$TMP"
  NPROC_PER_NODE=$NPROC \
  MAX_PIXELS=1003520 \
  swift infer \
      --model "$MERGED" \
      --infer_backend vllm \
      --val_dataset "$TODO" \
      --max_new_tokens 256 \
      --vllm_tensor_parallel_size 1 \
      --vllm_max_num_seqs $MAX_NUM_SEQS \
      --system 'You are a helpful assistant in natural image tampering detection.' \
      --result_path "$TMP"
  [ -f "$TMP" ] && cat "$TMP" >> "$RESULT" && rm -f "$TMP"

  after=$(wc -l < "$RESULT" 2>/dev/null || echo 0)
  if [ "$after" -le "$before" ]; then
    echo ">>> no progress this chunk (likely OOM inside one chunk)."
    echo ">>> lower the chunk/batch and re-run, e.g.:  CHUNK=$((CHUNK/2)) MAX_NUM_SEQS=32 NPROC=4 bash $0 $TOOL $CKPT"
    exit 1
  fi
  echo ">>> progress: $after samples written so far"
done
rm -f "$TODO"
echo ">>> Done. Results at: $RESULT"
