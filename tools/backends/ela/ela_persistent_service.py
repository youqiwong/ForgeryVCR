import os
import sys
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# [START] 完全复制您提供的 ELATool 类逻辑
# ==========================================
class ELATool:
    def __init__(self, quality=90, scale=15):
        """
        初始化 ELA 工具
        :param quality: 重压缩的 JPEG 质量 (通常 90 或 95)
        :param scale: 差分放大的倍数 (用于增强原始信号)
        """
        self.quality = quality
        self.scale = scale

    def process(self, image_path):
        """
        生成 ELA 热力图
        """
        if not os.path.exists(image_path):
            print(f"[Error] Image not found: {image_path}")
            return None

        # 1. 读取原图
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            print(f"[Error] Failed to load image: {image_path}")
            return None

        # 2. 模拟 JPEG 重压缩 (Re-compression)
        # 使用 cv2.imencode/imdecode 在内存中完成，无需写盘
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        success, encimg = cv2.imencode('.jpg', orig_img, encode_param)
        if not success:
            return None
        resaved_img = cv2.imdecode(encimg, 1)

        # 3. 计算绝对差分 (Absolute Difference)
        diff = cv2.absdiff(orig_img, resaved_img)

        # 4. 转为灰度 (Grayscale)
        # ELA 误差通常在三个通道都有，取最大值或平均值均可。
        # 这里转为灰度图，代表“误差强度”
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 5. 初始放大与截断 (Basic Scaling)
        # ELA 信号通常很微弱，先乘一个倍数 (Scale)
        diff_scaled = diff_gray.astype(np.float32) * self.scale
        diff_scaled = np.clip(diff_scaled, 0, 255).astype(np.uint8)
        
        # 保存一份基础增强的图 (Raw Map，类似传统 ELA 结果)
        raw_map = diff_scaled

        # 6. 视觉增强 (Visual Enhancement - CLAHE)
        # 这一步是为了对齐 SRM/CFA 的视觉风格
        # 提高对比度，让微弱的压缩痕迹差异更明显
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_map = clahe.apply(raw_map)

        # 7. 转为伪彩色热力图 (Heatmap)
        # 蓝色=低误差(原图压缩痕迹一致)，红色=高误差(被篡改/拼贴区域)
        heatmap = cv2.applyColorMap(enhanced_map, cv2.COLORMAP_JET)

        return heatmap, enhanced_map, raw_map
# ==========================================
# [END] 类逻辑结束
# ==========================================

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'tool': 'ela'})

@app.route('/infer', methods=['POST'])
def infer():
    if not request.json or 'image_path' not in request.json or 'output_path' not in request.json:
        return jsonify({'status': 'error', 'message': 'Missing image_path or output_path'}), 400

    image_path = request.json['image_path']
    output_path = request.json['output_path']
    
    # 保持默认值与您的 main 函数一致
    quality = request.json.get('quality', 90)
    scale = request.json.get('scale', 50)

    try:
        # 实例化工具
        tool = ELATool(quality=quality, scale=scale)
        
        # 调用 process
        result = tool.process(image_path)
        
        if result is not None:
            # 解包返回的三个结果
            heatmap, enhanced_map, raw_map = result
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # API 服务只负责保存最终的热力图 (Heatmap)
            # 如果您后续需要保存其他图，可以在这里添加逻辑，但通常 API 只返回一个主结果
            cv2.imwrite(output_path, heatmap)
            
            return jsonify({
                'status': 'success', 
                'output_path': output_path
            })
        else:
            return jsonify({'status': 'error', 'message': 'ELA processing returned None (Image load fail or encode fail)'}), 500
            
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("Starting ELA Heatmap Service (Exact Logic) on port 5003...")
    app.run(host='0.0.0.0', port=5003)
