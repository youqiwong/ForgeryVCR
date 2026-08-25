import os
import argparse
import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np

# --- 参数解析 ---
parser = argparse.ArgumentParser(description="FFT Heatmap Multi-GPU Inference Service Instance")
parser.add_argument('--port', type=int, required=True, help='Port to run this service instance on.')
args = parser.parse_args()
PORT = args.port

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - Port {PORT} - %(levelname)s - %(message)s')

# --- 全局配置 ---
HIGH_PASS_RADIUS = 30 
logging.info(f"[FFT Heatmap Service] Initialized with RADIUS = {HIGH_PASS_RADIUS}")

# --- 核心处理逻辑 (基于提供的 Unit Test) ---
def run_fft_inference(image_path: str, output_path: str):
    """
    生成 FFT 高频热力图 (Mag -> Norm -> CLAHE -> JET)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_color = cv2.imread(image_path)
    if img_color is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # 1. 准备高通滤波器掩码 (中心为0)
    rows, cols, _ = img_color.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= HIGH_PASS_RADIUS * HIGH_PASS_RADIUS
    mask[mask_area] = 0

    processed_channels = []
    for channel in cv2.split(img_color):
        # DFT
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shifted = np.fft.fftshift(dft)
        
        # 应用掩码 (高通滤波)
        fshift_filtered = dft_shifted * mask
        
        # 逆变换
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back_channel = cv2.idft(f_ishift)
        
        # 计算幅值
        magnitude = cv2.magnitude(img_back_channel[:, :, 0], img_back_channel[:, :, 1])
        
        # --- [核心逻辑] ---
        # 直接将幅值归一化到 0-255
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        processed_channels.append(norm_mag.astype(np.uint8))

    # 合并通道
    raw_map = cv2.merge(processed_channels)

    # 转为灰度以计算总能量
    energy = cv2.cvtColor(raw_map, cv2.COLOR_BGR2GRAY)

    # 视觉增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_map = clahe.apply(energy)

    # 转为热力图 (Heatmap: Blue->Red)
    heatmap = cv2.applyColorMap(enhanced_map, cv2.COLORMAP_JET)

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, heatmap)
    
    return output_path

# --- Flask 应用 ---
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'tool': 'fft'})

@app.route('/infer', methods=['POST'])
def infer():
    if 'image_path' not in request.json or 'output_path' not in request.json:
        return jsonify({'status': 'error', 'message': 'Missing parameters: image_path or output_path'}), 400

    image_path = request.json['image_path']
    output_path = request.json['output_path']
    
    try:
        result_path = run_fft_inference(image_path, output_path)
        return jsonify({'status': 'success', 'output_path': result_path})
            
    except Exception as e:
        logging.error(f"Inference error on {image_path}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logging.info(f"[FFT Heatmap Service] Starting Flask server on port {PORT}.")
    app.run(host='0.0.0.0', port=PORT)
