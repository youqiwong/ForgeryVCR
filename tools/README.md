# Forensic Tool Service

The forensic operators (ELA, FFT, NPP) run as **persistent GPU micro-services** behind a
lightweight **Flask API gateway**. The data-generation stage (`data/`) and, indirectly, the
tool-map indices used at inference all obtain their forensic maps through this gateway.

```
 caller ──POST /run/<tool>──▶  tool_api_server.py (gateway, :5000)
                                     │  round-robin over the tool's worker pool
                                     ▼
                         backends/<tool>/*_persistent_service.py  (one worker per GPU)
```

## Bundled backends

| Tool | Meaning | Port(s) | Type | Backend |
|------|---------|---------|------|---------|
| **gateway** | request router | `5000` | single | `tool_api_server.py` |
| **ela** | Error Level Analysis (compression) | `5003` | single | `backends/ela/` |
| **npp** | grayscale Noiseprint++ fingerprint | `5010–5017` | configurable pool | `backends/npp/npp_persistent_service_multi.py` |
| **fft** | frequency-domain analysis | `5020–5027` | configurable pool | `backends/fft/` |

The gateway and bundled services use ports in the `5000–5027` range. Pooled
services occupy up to eight consecutive ports, one worker per device.

`backends/npp/` also vendors its minimal runtime dependencies — `npp_service.py`,
`lib/` (the DnCNN / noiseprint++ definition), `dataset/dataset_test.py`, and the 2.2 MB
`pretrained_models/noiseprint++/noiseprint++.th` weight — so it runs standalone.
The code and weight in this backend remain subject to the upstream terms in
[`backends/npp/LICENSE.txt`](backends/npp/LICENSE.txt), including its
informational and nonprofit-use restriction.

## Start / stop

```bash
# from repo root
bash tools/start.sh
# stop: Ctrl+C (all workers are tracked and killed on exit)
```

Environment overrides:

| Var | Default | Meaning |
|-----|---------|---------|
| `PYTHON` | `python` | interpreter (e.g. `PYTHON=/usr/bin/python3.12`) |
| `INSTANCES` | `1` | workers per pooled service (one per GPU, valid range 1–8); the gateway reads the same value automatically |
| `LOG_DIR` | `tools/logs` | per-worker logs |

Each pooled worker is pinned to one accelerator device. If fewer than eight
devices are available, launch with, for example,
`INSTANCES=4 bash tools/start.sh`; no source edit is required.

## Health check

```bash
curl -fsS http://127.0.0.1:5000/health     # gateway and all configured workers
curl -fsS http://127.0.0.1:5003/health     # ELA
curl -fsS http://127.0.0.1:5010/health     # NPP worker 0
curl -fsS http://127.0.0.1:5020/health     # FFT worker 0
```

Or inspect `tools/logs/*.log` (watch that each NPP worker loads `noiseprint++.th`).
