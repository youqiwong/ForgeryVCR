#!/bin/bash
# ============================================================
# ForgeryVCR — Forensic Tool Service Launcher
# ------------------------------------------------------------
# Starts the forensic-tool micro-services + the API gateway (port 5000)
# that the data-generation and inference stages call.
#
# Core tools bundled in this repo (tools/backends/):
#   - NPP  (noise fingerprint) ports 5010-5017 (pool)
#   - FFT  (frequency)         ports 5020-5027 (pool)
#   - ELA  (compression)       port  5003       (single)
# Gateway: tools/tool_api_server.py         port  5000
#
# No tmux; everything runs in the background and is killed on Ctrl+C.
# ============================================================

set -u

# --- Resolve repo-relative paths (no hardcoded absolute paths) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKENDS="${SCRIPT_DIR}/backends"
PYTHON="${PYTHON:-python}"          # override with: PYTHON=/usr/bin/python3.12 bash start.sh
INSTANCES="${INSTANCES:-1}"         # parallel instances (one per GPU) for pooled services
if (( INSTANCES < 1 || INSTANCES > 8 )); then
    echo "[ERROR] INSTANCES must be between 1 and 8 to avoid port overlap." >&2
    exit 1
fi
export INSTANCES
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

PIDS=()

cleanup() {
    local status=$?
    trap - SIGINT SIGTERM EXIT
    echo
    echo "--- Stopping all services ---"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill "$pid"
        fi
    done
    echo "All services stopped."
    exit "${status}"
}
trap cleanup SIGINT SIGTERM EXIT

echo "--- Starting ForgeryVCR forensic services (logs in ${LOG_DIR}) ---"
echo "[INFO] python=${PYTHON}  instances=${INSTANCES}"
echo "[INFO] Press Ctrl+C to stop all services."

# --- Launch a pool of INSTANCES workers, one pinned per GPU ---
# Running by absolute script path puts the script's own dir on sys.path[0],
# so the NPP backend's local `lib` and `dataset` imports resolve.
launch_pool() {
    local NAME=$1 BASE_PORT=$2 SCRIPT=$3
    echo "[STARTING] ${NAME} (ports ${BASE_PORT}-$((BASE_PORT+INSTANCES-1)))"
    for (( i=0; i<INSTANCES; i++ )); do
        local PORT=$((BASE_PORT + i))
        env ROCM_VISIBLE_DEVICES=${i} CUDA_VISIBLE_DEVICES=${i} \
            "${PYTHON}" "${SCRIPT}" --port ${PORT} > "${LOG_DIR}/${NAME}_gpu${i}.log" 2>&1 &
        PIDS+=($!)
    done
}

# --- 1. Core pooled services ---------------------------------

# NPP (grayscale noise fingerprint, Noiseprint++)
launch_pool "npp" 5010 "${BACKENDS}/npp/npp_persistent_service_multi.py"

# FFT (frequency domain)
launch_pool "fft" 5020 "${BACKENDS}/fft/fft_persistent_service.py"

# --- 2. Core single-instance service -------------------------

# ELA (compression). Binds its own hardcoded port (5003); no --port arg.
echo "[STARTING] ela (port 5003)"
"${PYTHON}" "${BACKENDS}/ela/ela_persistent_service.py" \
    > "${LOG_DIR}/ela.log" 2>&1 &
PIDS+=($!)

# --- 3. API gateway (port 5000) ------------------------------
echo "[STARTING] API gateway (port 5000)"
"${PYTHON}" "${SCRIPT_DIR}/tool_api_server.py" > "${LOG_DIR}/api_gateway.log" 2>&1 &
PIDS+=($!)

wait_for_health() {
    local URL=$1
    local ATTEMPTS=${2:-180}
    for (( attempt=1; attempt<=ATTEMPTS; attempt++ )); do
        if "${PYTHON}" -c \
            'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3).read()' \
            "${URL}" >/dev/null 2>&1; then
            return 0
        fi
        for pid in "${PIDS[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "[ERROR] a forensic service exited during startup; inspect ${LOG_DIR}/*.log" >&2
                return 1
            fi
        done
        sleep 1
    done
    echo "[ERROR] forensic services did not become ready: ${URL}" >&2
    return 1
}

echo "[WAITING] loading forensic services..."
if ! wait_for_health "http://127.0.0.1:5000/health"; then
    exit 1
fi

echo
echo "--- All services ready (total processes: ${#PIDS[@]}) ---"
echo "Keeping script alive. Press Ctrl+C to shut down."
wait
