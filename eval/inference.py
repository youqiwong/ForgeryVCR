# run_inference.py
import os
import sys
import json
import re
import argparse
import random
import traceback
import copy
import multiprocessing as mp
import queue
import numpy as np
import torch
from PIL import Image, ImageFile
from termcolor import cprint
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple

# 导入配置
import config_inference as cfg

# ==================== 工具类与函数 (Agent) ====================

def setup_worker_environment(gpu_id):
    """设置 Worker 的 GPU 环境"""
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['ROCM_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

def load_pil_image(img_input):
    """加载 PIL 图像，处理路径或 Image 对象"""
    try:
        if isinstance(img_input, str):
            if os.path.exists(img_input):
                return Image.open(img_input).convert('RGB')
        elif isinstance(img_input, Image.Image):
            return img_input.convert('RGB')
        return None
    except Exception:
        return None

def sanitize_bbox(bbox: List[int], img_w: int, img_h: int) -> Tuple[List[int], bool]:
    """检查并修正 BBox，防止越界或畸形"""
    is_valid = True
    if not isinstance(bbox, list) or len(bbox) != 4:
        is_valid = False
    else:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0: is_valid = False
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h: is_valid = False
            if is_valid:
                aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 999
                if aspect_ratio > 200: is_valid = False
        except (ValueError, TypeError):
            is_valid = False

    if is_valid:
        return [x1, y1, x2, y2], False
    else:
        # Fallback 到中心区域
        cx, cy = img_w // 2, img_h // 2
        half_w = max(1, img_w // 4) 
        half_h = max(1, img_h // 4)
        fallback_bbox = [max(0, cx - half_w), max(0, cy - half_h), min(img_w, cx + half_w), min(img_h, cy + half_h)]
        return fallback_bbox, True

class ToolMapLoader:
    """加载 ELA/NPP/FFT 等工具生成的预处理图映射"""
    def __init__(self, json_paths_config):
        self.tool_maps = {name: self._load_and_merge_json(paths) for name, paths in json_paths_config.items()}
    
    def _load_and_merge_json(self, paths):
        if isinstance(paths, str): paths = [paths]
        merged_map = {}
        for path in paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f: 
                        data = json.load(f)
                        json_dir = os.path.dirname(os.path.abspath(path))
                        def resolve(value):
                            if not value or os.path.isabs(value):
                                return value
                            return os.path.abspath(os.path.join(json_dir, value))
                        # 建立从原图路径到工具图路径的映射
                        current_map = {
                            resolve(item['image_path']): resolve(item['tool_map_path'])
                            for item in data
                        }
                        merged_map.update(current_map)
            except Exception as e:
                print(f"Error loading tool map json {path}: {e}")
        return merged_map
    
    def get_tool_map_path(self, tool_name, original_image_path):
        if tool_name not in self.tool_maps: return None
        tool_map = self.tool_maps[tool_name]
        # 优先全路径匹配
        if original_image_path in tool_map: return tool_map[original_image_path]
        # 其次文件名匹配
        basename = os.path.basename(original_image_path)
        for img_path, map_path in tool_map.items():
            if os.path.basename(img_path) == basename: return map_path
        return None

def parse_tool_calls(content: str):
    """从模型输出中解析工具调用"""
    tool_call_str_list = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
    if not tool_call_str_list: return None
    parsed_tool_calls = []
    for tool_call_str in tool_call_str_list:
        try:
            tool_data = json.loads(tool_call_str)
            parsed_tool_calls.append(tool_data)
        except json.JSONDecodeError:
            name_match = re.search(r'["\']name["\']\s*:\s*["\'](.*?)["\']', tool_call_str)
            if name_match:
                parsed_tool_calls.append({"name": name_match.group(1), "arguments": {}})
    return parsed_tool_calls if parsed_tool_calls else None

def check_answer_pair(content: str):
    return len(re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)) >= 1

# ==================== Agent 推理逻辑 ====================

def run_batch_agent_inference(engine, batch_items, request_config_base, tools, tool_map_loader):
    from swift.llm import InferRequest
    MAX_TURNS = 5
    states = []
    
    # 初始化状态
    for item in batch_items:
        image_path = item.get("images", [None])[0]
        initial_prompt_text = item.get("_eval_user_prompt")
        if initial_prompt_text is None:
            initial_prompt_text = random.choice(cfg.USER_PROMPTS)
        state = {
            "original_item": item,
            "image_path": image_path,
            "messages": [{"role": "user", "content": f"<image>{initial_prompt_text}"}],
            "images": [image_path],
            # 用于保存最终 JSON 的对话记录
            "dialogue_for_json": [{"role": "user", "content": initial_prompt_text, "image_path": os.path.abspath(image_path) if image_path else ""}],
            "done": False,
            "error": None
        }
        if not image_path or not os.path.exists(image_path):
            state["done"] = True
            state["error"] = "Image path not found or invalid"
        states.append(state)

    # 多轮对话循环
    for turn in range(1, MAX_TURNS + 1):
        active_states = [s for s in states if not s["done"]]
        if not active_states: break
        
        requests = []
        for s in active_states:
            req = InferRequest(messages=s["messages"], images=s["images"], tools=tools)
            requests.append(req)
        
        try:
            resp_list = engine.infer(requests, request_config_base)
        except Exception as e:
            for s in active_states:
                s["error"] = f"Batch inference error at turn {turn}: {e}"
                s["done"] = True
            break

        for s, resp in zip(active_states, resp_list):
            content_str = resp.choices[0].message.content or ""
            
            s["dialogue_for_json"].append({"role": "assistant", "content": content_str, "turn": turn})
            s["messages"].append({"role": "assistant", "content": content_str})
            
            # 检查是否结束 (输出 <answer>)
            if check_answer_pair(content_str):
                s["done"] = True
                continue
            
            # 解析工具调用
            parsed_tool_calls = parse_tool_calls(content_str)
            # 兼容 vllm tool_calls 字段
            if not parsed_tool_calls and resp.choices[0].message.tool_calls:
                parsed_tool_calls = []
                for tc in resp.choices[0].message.tool_calls:
                    args = {}
                    if tc.function.arguments:
                        try: args = json.loads(tc.function.arguments)
                        except: pass
                    parsed_tool_calls.append({"name": tc.function.name, "arguments": args})

            if not parsed_tool_calls:
                s["done"] = True
                continue
                
            # 执行工具
            tool_call = parsed_tool_calls[0]
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('arguments', {})
            tool_image_result = None
            status = "success"
            err_msg = ""
            
            if tool_name == 'zoom_in':
                bbox = tool_args.get('bbox')
                try:
                    with load_pil_image(s["image_path"]) as pil_img:
                        if pil_img is None: raise ValueError("Failed to reload original image")
                        w, h = pil_img.size
                        final_bbox, is_fallback = sanitize_bbox(bbox, w, h)
                        if is_fallback: s["dialogue_for_json"].append({"warning": f"Invalid bbox {bbox} corrected to {final_bbox}"})
                        crop_img = pil_img.crop((final_bbox[0], final_bbox[1], final_bbox[2], final_bbox[3]))
                        tool_image_result = crop_img.copy()
                except Exception as e:
                    status = "error"
                    err_msg = f"Zoom failed: {str(e)}"
            elif tool_name in ['ELA', 'FFT', 'NPP']:
                map_path = tool_map_loader.get_tool_map_path(tool_name, s["image_path"])
                if map_path and os.path.exists(map_path): tool_image_result = map_path
                else:
                    status = "error"
                    err_msg = f"Map not found for {tool_name}"
            else:
                status = "error"
                err_msg = f"Unknown tool: {tool_name}"

            # 构造工具返回消息
            if status == "success" and tool_image_result is not None:
                tool_response_content = json.dumps({"status": "success", "tool_image": "<image>"})
                s["messages"].append({"role": "tool_response", "content": tool_response_content})
                s["images"].append(tool_image_result)
                log_img_val = tool_image_result if isinstance(tool_image_result, str) else "PIL_IMAGE_OBJECT"
                s["dialogue_for_json"].append({"role": "tool_response", "content": tool_response_content, "tool_name": tool_name, "tool_args": tool_args, "image_info": log_img_val, "turn": turn})
            else:
                fail_content = json.dumps({"status": "error", "message": err_msg})
                s["messages"].append({"role": "tool_response", "content": fail_content})
                s["dialogue_for_json"].append({"role": "tool_response", "content": fail_content, "error": err_msg, "turn": turn})
    
    results = []
    for s in states:
        final_item = s["original_item"].copy()
        final_item.pop("_eval_user_prompt", None)
        final_item['messages'] = s["dialogue_for_json"]
        if s["error"]: final_item['inference_error'] = s["error"]
        results.append(final_item)
    return results

def agent_worker_process(gpu_id: int, data_chunk: list, output_queue: mp.Queue,
                         model_path: str, batch_size: int, seed: int):
    try:
        setup_worker_environment(gpu_id)
        worker_seed = seed + gpu_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        # 延迟导入以避免主进程冲突
        from swift.llm import VllmEngine, RequestConfig
        from swift.plugin import agent_templates
        
        cprint(f"[Agent Worker {gpu_id}] Loading VllmEngine: {model_path}", 'yellow')
        
        engine = VllmEngine(
            model_id_or_path=model_path,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": 5},
            seed=seed,
        )
        if hasattr(engine.default_template, 'agent_template'):
            engine.default_template.agent_template = agent_templates['hermes']()
        
        cprint(f"[Agent Worker {gpu_id}] Engine Ready.", 'green')
        tool_map_loader = ToolMapLoader(cfg.TOOL_MAP_JSON_PATHS)
        request_config = RequestConfig(
            max_tokens=3072,
            temperature=1.0,
            stop=[],
            seed=seed,
        )
        tools = json.loads(cfg.TOOLS_JSON_STRING)
        
        for i in tqdm(range(0, len(data_chunk), batch_size), desc=f"GPU {gpu_id} Inference", position=gpu_id):
            batch_items = data_chunk[i : i + batch_size]
            try:
                batch_results = run_batch_agent_inference(engine, batch_items, request_config, tools, tool_map_loader)
                output_queue.put(batch_results) 
            except Exception as e:
                err_results = []
                for item in batch_items:
                    itm = item.copy()
                    itm['inference_error'] = str(e)
                    err_results.append(itm)
                output_queue.put(err_results)
                cprint(f"[Agent Worker {gpu_id}] Batch Error: {e}", 'red')
                cprint(traceback.format_exc(), 'red')

        output_queue.put(None) # 结束信号
    except Exception as e:
        cprint(f"[Agent Worker {gpu_id}] FATAL ERROR: {e}", 'red')
        cprint(traceback.format_exc(), 'red')
        output_queue.put({"_worker_fatal": gpu_id, "error": repr(e)})
        output_queue.put(None)

# ==================== SAM2 相关函数 ====================

def parse_bbox_from_answer(answer_text):
    """从 <answer> 中提取 BBox 并还原到图像坐标"""
    if not answer_text: return []
    # 匹配 (x1, y1), (x2, y2) 格式
    pattern = re.compile(r'\((\d+)\s*,\s*(-?\d+)\)\s*,\s*\((\d+)\s*,\s*(-?\d+)\)')
    matches = pattern.findall(answer_text)
    bboxes = []
    for match in matches:
        try:
            x1_raw, y1_raw, x2_raw, y2_raw = map(int, match)
            # 假设输出是 1000x1000 归一化的，还原到 cfg.IMG_SIZE
            scale = cfg.IMG_SIZE / 1000.0
            x1 = int(x1_raw * scale)
            y1 = int(y1_raw * scale)
            x2 = int(x2_raw * scale)
            y2 = int(y2_raw * scale)
            bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        except: 
            continue
    return bboxes

def get_grid_points(bbox, grid_size=2):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    points = []
    step_x, step_y = w / (grid_size + 1), h / (grid_size + 1)
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            points.append([x1 + i * step_x, y1 + j * step_y])
    labels = [1] * len(points)
    return np.array(points), np.array(labels)

def sam2_worker_process(gpu_queue, task_queue, result_queue, mode, mask_output_root):
    """SAM2 独立 Worker，用于生成掩码"""
    try:
        # 避免 matplotlib 后端问题
        import matplotlib
        matplotlib.use('Agg')
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        gpu_id = gpu_queue.get()
        os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['ROCM_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # The physical GPU is isolated by the visibility variables, so it is
        # always logical device 0 inside this worker.
        device = "cuda:0"
        
        # 加载 SAM2 模型
        torch.set_grad_enabled(False)
        sam2_model = build_sam2(cfg.SAM2_CONFIG_NAME, cfg.SAM2_CHECKPOINT, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        while True:
            task = task_queue.get()
            if task is None:
                break # 结束信号
            
            try:
                dataset_name = task['dataset_name']
                image_path = task['image_path']
                pred_bboxes = task['pred_bboxes']
                
                if not os.path.exists(image_path):
                    result_queue.put(("error", image_path, "input image not found"))
                    continue
                if not pred_bboxes:
                    result_queue.put(("error", image_path, "no valid predicted bounding boxes"))
                    continue
                
                image_pil = Image.open(image_path).convert("RGB")
                w, h = image_pil.size
                
                # 设置图像
                predictor.set_image(image_pil)
                
                pred_mask_final = np.zeros((h, w), dtype=bool)
                
                # 处理每个 bbox
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for bbox in pred_bboxes:
                        safe_box = [max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])]
                        if safe_box[2] <= safe_box[0] or safe_box[3] <= safe_box[1]: continue
                        
                        points, labels = get_grid_points(safe_box, grid_size=2) # 默认使用 grid 模式
                        
                        # 调用 SAM2
                        masks, scores, _ = predictor.predict(
                            point_coords=points,
                            point_labels=labels,
                            box=np.array(safe_box),
                            multimask_output=False
                        )
                        current_mask = masks[0].astype(bool)
                        pred_mask_final = np.logical_or(pred_mask_final, current_mask)
                
                # 保存掩码
                save_dir = os.path.join(mask_output_root, dataset_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(image_path))[0] + ".png")
                
                mask_img = Image.fromarray((pred_mask_final * 255).astype(np.uint8))
                mask_img.save(save_path)
                result_queue.put(("success", image_path, save_path))
                
            except Exception as e:
                image_path = task.get('image_path', '<unknown>')
                cprint(f"[SAM2 Worker] Error on {image_path}: {e}", 'red')
                result_queue.put(("error", image_path, repr(e)))
                
        gpu_queue.put(gpu_id) # 归还 GPU ID (虽然进程结束了，但为了逻辑完整)
    except Exception as e:
        cprint(f"[SAM2 Worker] Init Error: {e}", 'red')
        result_queue.put(("fatal", "<worker-init>", repr(e)))

# ==================== 主流程控制 ====================

def run_agent_inference(args):
    """步骤 1: 运行 Agent 推理生成 JSON"""
    cprint(f"\n>>> Starting Agent Inference...", 'cyan')
    cprint(f"Model: {args.model_path}", 'magenta')
    cprint(f"Output: {args.output_dir}", 'magenta')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载数据
    full_dataset = []
    if os.path.exists(cfg.CONSOLIDATED_DATASET_PATH):
        with open(cfg.CONSOLIDATED_DATASET_PATH, 'r', encoding='utf-8') as f:
            full_dataset = json.load(f)
        dataset_dir = os.path.dirname(os.path.abspath(cfg.CONSOLIDATED_DATASET_PATH))
        for item in full_dataset:
            item["images"] = [
                p if os.path.isabs(p) else os.path.abspath(os.path.join(dataset_dir, p))
                for p in item.get("images", [])
            ]
    else:
        cprint(f"Dataset not found: {cfg.CONSOLIDATED_DATASET_PATH}", 'red')
        sys.exit(1)
        
    # 2. 筛选数据集
    target_datasets = args.datasets if 'all' not in args.datasets else cfg.DEFAULT_TARGET_DATASETS
    # 处理空列表情况，默认为所有
    if not target_datasets: target_datasets = cfg.DEFAULT_TARGET_DATASETS
    
    target_items = [item for item in full_dataset if item.get('dataset') in target_datasets]
    cprint(f"Loaded {len(target_items)} samples for datasets: {target_datasets}", 'green')
    
    if len(target_items) == 0:
        cprint("No samples found. Exiting.", 'yellow')
        return

    # Assign prompts in the parent process so prompt selection is deterministic
    # and independent of GPU count / worker scheduling.
    prompt_rng = random.Random(args.seed)
    for item in target_items:
        item["_eval_user_prompt"] = prompt_rng.choice(cfg.USER_PROMPTS)

    # 3. 启动多进程 Agent 推理
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No accelerator device is visible for agent inference.")
    batch_size = 64
    chunk_size = (len(target_items) + num_gpus - 1) // num_gpus
    chunks = [target_items[i:i + chunk_size] for i in range(0, len(target_items), chunk_size)]
    
    output_queue = mp.Queue()
    processes = []
    
    for i in range(min(num_gpus, len(chunks))):
        p = mp.Process(target=agent_worker_process, 
                       args=(i, chunks[i], output_queue, args.model_path, batch_size, args.seed))
        p.start()
        processes.append(p)
        
    # 4. 收集结果
    all_results = []
    worker_failures = []
    finished_count = 0
    while finished_count < len(processes):
        try:
            res = output_queue.get(timeout=5)
            if res is None:
                finished_count += 1
            elif isinstance(res, dict) and "_worker_fatal" in res:
                worker_failures.append(res)
            else:
                all_results.extend(res)
        except queue.Empty:
            if not any(p.is_alive() for p in processes):
                break
            
    for p in processes:
        p.join()

    crashed = [p.exitcode for p in processes if p.exitcode != 0]
    inference_errors = [item for item in all_results if item.get("inference_error")]
    if (worker_failures or crashed or inference_errors
            or len(all_results) != len(target_items)):
        raise RuntimeError(
            "Agent inference incomplete: "
            f"results={len(all_results)}/{len(target_items)}, "
            f"sample_errors={len(inference_errors)}, "
            f"worker_failures={worker_failures}, worker_exitcodes={crashed}"
        )
    
    # 5. 保存 JSON 结果
    results_by_dataset = defaultdict(list)
    for item in all_results:
        results_by_dataset[item.get("dataset", "unknown")].append(item)
        
    for ds_name, items in results_by_dataset.items():
        out_path = os.path.join(args.output_dir, f"{ds_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            
    cprint("[Agent Inference] Done. Results saved.", 'green')

def run_sam2_generation(args):
    """步骤 2: 运行 SAM2 生成掩码"""
    cprint(f"\n>>> Starting SAM2 Mask Generation...", 'cyan')

    try:
        import sam2  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAM2 is not installed in the current Python environment.\n"
            "From the repository root run:\n"
            "  git clone https://github.com/facebookresearch/sam2.git third_party/sam2\n"
            "  SAM2_BUILD_CUDA=0 python -m pip install "
            "--no-build-isolation -e ./third_party/sam2"
        ) from exc

    if not os.path.isfile(cfg.SAM2_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {cfg.SAM2_CHECKPOINT}\n"
            "Set SAM2_CHECKPOINT=/absolute/path/to/sam2_hiera_large.pt"
        )
    
    mask_output_root = os.path.join(args.output_dir, "pred_masks")
    cprint(f"Mask Output: {mask_output_root}", 'magenta')
    os.makedirs(mask_output_root, exist_ok=True)
    
    # 1. 读取刚刚生成的 JSON 结果
    tasks = []
    target_datasets = args.datasets if 'all' not in args.datasets and args.datasets else cfg.DEFAULT_TARGET_DATASETS
    
    for ds_name in target_datasets:
        json_path = os.path.join(args.output_dir, f"{ds_name}.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
            
        for sample in samples:
            # 只处理被判断为 fake 的图片
            # 这里简单判断：如果有 answer 且包含 bbox，或者 explicitly is_fake (取决于你的数据结构)
            # 我们根据 answer 内容解析 bbox
            last_msg = None
            if 'messages' in sample:
                for m in reversed(sample['messages']):
                    if m.get('role') == 'assistant':
                        last_msg = m
                        break
            
            if last_msg:
                content = last_msg.get('content', '')
                bboxes = parse_bbox_from_answer(content)
                if bboxes:
                    tasks.append({
                        'dataset_name': ds_name,
                        'image_path': sample.get('images', [''])[0],
                        'pred_bboxes': bboxes
                    })
    
    cprint(f"Found {len(tasks)} samples requiring mask generation.", 'green')
    if not tasks: return

    # 2. 启动 SAM2 Worker
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No accelerator device is visible for SAM2 generation.")
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue() # 这里的逻辑里不需要返回结果，只保存图片
    
    for i in range(num_gpus): gpu_queue.put(i)
    for t in tasks: task_queue.put(t)
    for _ in range(num_gpus): task_queue.put(None) # 终止信号
    
    workers = []
    # 使用 Spawn 模式启动
    ctx = mp.get_context('spawn')
    for _ in range(num_gpus):
        p = ctx.Process(target=sam2_worker_process, args=(gpu_queue, task_queue, result_queue, 'grid', mask_output_root))
        p.start()
        workers.append(p)
        
    for p in workers:
        p.join()

    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    failures = [result for result in results if result[0] != "success"]
    crashed = [p.exitcode for p in workers if p.exitcode != 0]
    success_count = sum(result[0] == "success" for result in results)
    if failures or crashed or success_count != len(tasks):
        details = "; ".join(f"{path}: {message}" for _, path, message in failures[:5])
        raise RuntimeError(
            "SAM2 mask generation incomplete: "
            f"success={success_count}/{len(tasks)}, worker_exitcodes={crashed}. "
            f"{details}"
        )
    cprint("[SAM2 Generation] Done.", 'green')

if __name__ == "__main__":
    # 设置启动方法，防止 CUDA 初始化错误
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Unified Agent Inference & SAM2 Mask Gen")
    
    # 通用参数
    parser.add_argument('--model_path', type=str, required=True, help="Path to the checkpoint (directory)")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output JSONs and Masks")
    parser.add_argument('--datasets', nargs='+', default=[], help="List of datasets to process (default: all defined in config)")
    parser.add_argument('--seed', type=int, default=cfg.SEED,
                        help=f"Prompt selection and vLLM sampling seed (default: {cfg.SEED})")
    
    # 功能开关
    parser.add_argument('--skip_agent', action='store_true', help="Skip agent inference, run SAM2 only (requires existing JSONs)")
    parser.add_argument('--use_sam2', action='store_true', help="Run SAM2 after agent inference to generate binary masks")
    
    args = parser.parse_args()
    
    # 1. Agent 推理
    if not args.skip_agent:
        run_agent_inference(args)
    else:
        cprint("Skipping Agent Inference as requested.", 'yellow')
        
    # 2. SAM2 掩码生成
    if args.use_sam2:
        run_sam2_generation(args)
