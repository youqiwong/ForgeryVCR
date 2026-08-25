"""Persistent multi-device service for the bundled NPP backend."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Capture the assigned device before importing the accelerator runtime.
EXPECTED_GPU_ID = os.getenv('ROCM_VISIBLE_DEVICES', os.getenv('CUDA_VISIBLE_DEVICES', 'N/A'))

import cv2
import torch
from flask import Flask, jsonify, request

SERVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICE_DIR))
from npp_service import load_npp_model, run_npp_inference

parser = argparse.ArgumentParser(description="NPP Multi-GPU Inference Service Instance")
parser.add_argument('--port', type=int, required=True, help='Port to run this service instance on.')
args = parser.parse_args()
PORT = args.port

logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - GPU {EXPECTED_GPU_ID} - Port {PORT} - %(levelname)s - %(message)s')

# Each worker must see exactly one accelerator device.
visible_devices = torch.cuda.device_count()
logging.info(f"[SELF-CHECK] Expected Physical GPU: {EXPECTED_GPU_ID}, PyTorch Visible Devices: {visible_devices}")

if visible_devices > 1:
    logging.critical(f"[FATAL ERROR] Isolation FAILED! Process can see {visible_devices} GPUs instead of 1.")
    logging.critical("Exiting immediately to protect GPU 0 from overload. (Exit Code 99)")
    time.sleep(1)
    sys.exit(99)

if visible_devices == 0:
    logging.critical("[FATAL ERROR] No GPUs visible to PyTorch! Check drivers or environment variables. (Exit Code 98)")
    sys.exit(98)

# The sole visible physical device is addressed as cuda:0 inside the worker.
DEVICE = torch.device("cuda:0")
logging.info(f"[SELF-CHECK] Isolation SUCCESS. Using internal device '{DEVICE}' which maps to physical GPU {EXPECTED_GPU_ID}.")

# Resolve the bundled checkpoint from this file, independent of the caller's cwd.
MODEL_FILE = SERVICE_DIR / "pretrained_models" / "noiseprint++" / "noiseprint++.th"
logging.info("[NPP Service] Starting model loading...")

try:
    MODEL = load_npp_model(str(MODEL_FILE), DEVICE)
    MODEL.eval()
    logging.info(f"[NPP Service] Model loaded and set to eval(). Ready on Port {PORT}.")
except Exception as e:
    logging.critical(f"[FATAL ERROR] Model loading failed: {e}", exc_info=True)
    sys.exit(1)

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'tool': 'npp', 'device': str(DEVICE)})

@app.route('/infer', methods=['POST'])
def infer():
    payload = request.get_json(silent=True) or {}
    if 'image_path' not in payload or 'output_path' not in payload:
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400

    image_path = payload['image_path']
    output_path = payload['output_path']

    if not os.path.isfile(image_path):
        return jsonify({'status': 'error', 'message': f'Image not found: {image_path}'}), 404

    try:
        with torch.no_grad():
            result_gray_np = run_npp_inference(MODEL, image_path, DEVICE)

        if result_gray_np is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            if not cv2.imwrite(output_path, result_gray_np):
                return jsonify({'status': 'error', 'message': 'Failed to write NPP output.'}), 500
            return jsonify({'status': 'success', 'output_path': output_path})
        else:
            return jsonify({'status': 'error', 'message': 'Inference failed.'}), 500

    except Exception as e:
        logging.error(f"Inference error on {image_path}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
