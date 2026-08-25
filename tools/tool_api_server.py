import os
from flask import Flask, request, jsonify
import logging
import time
import requests
from itertools import cycle
import threading

# 环境变量清理
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["no_proxy"] = "*" 
os.environ["NO_PROXY"] = "*"

# --- Bundled forensic services ---
ELA_PORT = 5003
NPP_BASE_PORT = 5010
FFT_BASE_PORT = 5020
POOLED_INSTANCES = int(os.environ.get("INSTANCES", "1"))
if not 1 <= POOLED_INSTANCES <= 8:
    raise ValueError("INSTANCES must be between 1 and 8 to avoid port overlap")

TOOL_SERVICES = {
    "ela": f"http://127.0.0.1:{ELA_PORT}/infer",
    "fft": [f"http://127.0.0.1:{FFT_BASE_PORT + i}/infer" for i in range(POOLED_INSTANCES)],
    "npp": [f"http://127.0.0.1:{NPP_BASE_PORT + i}/infer" for i in range(POOLED_INSTANCES)],
}

# 为所有多实例服务创建迭代器和锁
service_iterators = {
    tool: cycle(endpoints)
    for tool, endpoints in TOOL_SERVICES.items() if isinstance(endpoints, list)
}
service_locks = {
    tool: threading.Lock()
    for tool in service_iterators
}

app = Flask(__name__)
# 减少日志干扰
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/health', methods=['GET'])
def health():
    """Report gateway readiness only after every configured worker is reachable."""
    worker_status = {}
    all_ready = True
    for tool_name, endpoints in TOOL_SERVICES.items():
        endpoint_list = endpoints if isinstance(endpoints, list) else [endpoints]
        statuses = []
        for endpoint in endpoint_list:
            health_url = endpoint.rsplit('/infer', 1)[0] + '/health'
            try:
                response = requests.get(health_url, timeout=2)
                ready = response.ok
            except requests.RequestException:
                ready = False
            statuses.append({'url': health_url, 'ready': ready})
            all_ready = all_ready and ready
        worker_status[tool_name] = statuses
    return jsonify({
        'status': 'ready' if all_ready else 'starting',
        'workers': worker_status,
    }), 200 if all_ready else 503

@app.route('/run/<tool_name>', methods=['POST'])
def run_tool(tool_name):
    total_timer_start = time.perf_counter()
    
    if tool_name not in TOOL_SERVICES:
        return jsonify({'status': 'error', 'message': f'Unknown tool: {tool_name}'}), 400
    
    try:
        client_payload = request.json
        if not client_payload or 'image_path' not in client_payload or 'output_path' not in client_payload:
            return jsonify({'status': 'error', 'message': 'Missing image_path or output_path'}), 400
            
        input_image_path = client_payload['image_path']
        if not os.path.exists(input_image_path):
            return jsonify({'status': 'error', 'message': f'Input image not found: {input_image_path}'}), 404

        # --- 服务选择与负载均衡 ---
        service_url_or_list = TOOL_SERVICES[tool_name]
        if isinstance(service_url_or_list, list):
            with service_locks[tool_name]:
                service_url = next(service_iterators[tool_name])
        else:
            service_url = service_url_or_list

        # --- 转发请求 ---
        # 这里的 timeout 设置为 600s，适应大模型或慢速工具
        response = requests.post(service_url, json=client_payload, timeout=600)
        
        if response.status_code == 200:
            resp_data = response.json()
            # 如果后端没返回 output_path，就用请求里的
            final_output = resp_data.get('output_path', client_payload['output_path'])
            
            total_time = time.perf_counter() - total_timer_start
            logging.info(f"[{tool_name}] Success: {total_time:.2f}s")
            
            return jsonify({
                'status': 'success',
                'output_path': final_output,
                'total_time': total_time
            })
        else:
            err_msg = f"Service Error {response.status_code}: {response.text}"
            logging.error(f"[{tool_name}] {err_msg}")
            return jsonify({'status': 'error', 'message': err_msg}), 500

    except requests.exceptions.ConnectionError:
        msg = f"Connection refused connecting to internal service for {tool_name}. Is the worker running?"
        logging.error(msg)
        return jsonify({'status': 'error', 'message': msg}), 503
    except Exception as e:
        msg = f"Gateway Error: {str(e)}"
        logging.error(msg)
        return jsonify({'status': 'error', 'message': msg}), 500

if __name__ == '__main__':
    print("API Gateway starting on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
