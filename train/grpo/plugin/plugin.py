import base64
import io
import os
import re
import json
import datetime
import gc
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ================= 新增绘图库 =================
import matplotlib
# 设置后端为 Agg，防止在无显示器的服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ms-swift imports
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.orm import ORM, orms
from swift.utils import get_logger, get_dist_setting
from scipy.optimize import linear_sum_assignment

logger = get_logger()

# ================= 1. 配置路径与常量 =================
REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = Path(
    os.environ.get(
        "GRPO_TOOL_MAP_ROOT",
        str(REPO_ROOT / "datasets" / "train" / "tool_maps"),
    )
)


ELA_DIR = str(TOOLS_ROOT / "ELA")
FFT_DIR = str(TOOLS_ROOT / "FFT")
NPP_DIR = str(TOOLS_ROOT / "NPP")

# ================= 2. 辅助函数 =================

def extract_tag_content(tag: str, text: str) -> str:
    """
    尝试从文本中提取 <tag>content</tag>。
    """
    if not text: return None
    
    # 1. 标准闭合标签
    pattern = re.compile(f'<{tag}>(.*?)</{tag}>', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    
    # 2. 针对 tool_call 可能存在的无闭合标签情况
    if tag == 'tool_call':
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        if start_tag in text:
            start_idx = text.find(start_tag) + len(start_tag)
            end_idx = text.find(end_tag)
            if end_idx != -1:
                return text[start_idx:end_idx].strip()
            else:
                return text[start_idx:].strip()
    return None

def parse_tool_call_from_text(text: str) -> Dict[str, Any]:
    """
    更强壮的工具调用解析函数：
    1. 尝试提取 <tool_call> 标签内容
    2. 如果失败，尝试通过正则直接匹配 JSON 结构
    """
    if not text: return None
    
    json_str = extract_tag_content('tool_call', text)
    
    # 如果没找到标签，尝试直接用正则找 JSON 结构
    if not json_str:
        try:
            # 这是一个简单的启发式正则，匹配 {"name": ..., "arguments": { ... }}
            json_pattern = re.compile(r'\{[\s\n]*"name"[\s\n]*:[\s\n]*".*?"[\s\n]*,[\s\n]*"arguments"[\s\n]*:[\s\n]*\{.*?\}[\s\n]*\}', re.DOTALL)
            match = json_pattern.search(text)
            if match:
                json_str = match.group(0)
        except:
            pass

    if not json_str:
        return None

    try:
        # 清理可能存在的 markdown 代码块标记
        clean_json = json_str.strip()
        clean_json = re.sub(r'^```json\s*', '', clean_json)
        clean_json = re.sub(r'^```\s*', '', clean_json)
        clean_json = re.sub(r'\s*```$', '', clean_json)
        
        return json.loads(clean_json.strip())
    except Exception:
        return None

def load_pil_image(img: Any) -> Image.Image:
    try:
        if isinstance(img, Image.Image):
            return img.convert('RGB')
        elif isinstance(img, dict):
            if 'bytes' in img and img['bytes']:
                return Image.open(io.BytesIO(img['bytes'])).convert('RGB')
            elif 'path' in img and img['path']:
                return Image.open(img['path']).convert('RGB')
        elif isinstance(img, str):
            if os.path.exists(img):
                return Image.open(img).convert('RGB')
            if ',' in img:
                img_data = img.split(',')[1]
            else:
                img_data = img
            try:
                img_bytes = base64.b64decode(img_data)
                return Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except:
                return None
        elif isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(img)).convert('RGB')
        elif hasattr(img, 'read'):
            return Image.open(img).convert('RGB')
        else:
            return img
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None

def get_original_image_path(infer_request: Any) -> str:
    if not infer_request.images:
        return None
    original_img = infer_request.images[0]
    if isinstance(original_img, str):
        if os.path.exists(original_img): return original_img
        return None
    if isinstance(original_img, Image.Image):
        if hasattr(original_img, 'filename') and original_img.filename: return original_img.filename
        if hasattr(original_img.fp, 'name'): return original_img.fp.name
        return None
    if isinstance(original_img, dict):
        return original_img.get('path')
    return None

def get_tool_image_path(original_image_path: str, tool_name: str) -> str:
    if not original_image_path: return None
    filename_with_ext = os.path.basename(original_image_path)
    filename_stem = os.path.splitext(filename_with_ext)[0]
    dataset_name = next(
        (
            parent.name
            for parent in Path(original_image_path).parents
            if parent.name.lower().endswith(("_tp", "_au"))
        ),
        None,
    )
    candidates = []
    if tool_name == 'ELA':
        if dataset_name:
            candidates.append(
                str(TOOLS_ROOT / dataset_name / "ELA" / f"{filename_stem}_ELA.png")
            )
        candidates.append(os.path.join(ELA_DIR, f"{filename_stem}_ELA.png"))
    elif tool_name == 'FFT':
        if dataset_name:
            candidates.append(
                str(TOOLS_ROOT / dataset_name / "FFT" / f"{filename_stem}_FFT.png")
            )
        candidates.append(os.path.join(FFT_DIR, f"{filename_stem}_FFT.png"))
    elif tool_name == 'NPP':
        if dataset_name:
            candidates.append(
                str(TOOLS_ROOT / dataset_name / "NPP" / f"{filename_stem}_NPP.png")
            )
        candidates.append(os.path.join(NPP_DIR, f"{filename_stem}_NPP.png"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""

# ================= 3. 奖励函数 (修正：遍历历史统计工具 + 绘图坐标轴) =================

class ForgeryDetectionReward(ORM):
    def __init__(self, frequency: int = 1):
        super().__init__()
        dist_info = get_dist_setting()
        self.rank = dist_info[2]
        self.call_count = 0
        self.frequency = frequency
        self.is_logging_process = (self.rank == 8) 
        self.plot_interval = max(
            1, int(os.environ.get("GRPO_PLOT_INTERVAL", "20"))
        )
        
        self.smooth_weight = 0.9
        
        logger.info(f"[Rank {self.rank}] Init ForgeryDetectionReward (Fixed Plots).")
        if self.is_logging_process:
            TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            BASE_LOG_DIR = os.environ.get(
                "GRPO_REWARD_LOG_DIR",
                str(REPO_ROOT / "train" / "grpo" / "reward_logs"),
            )
            self.output_base_dir = os.path.join(BASE_LOG_DIR, f'run_{TIMESTAMP}')
            
            self.metrics_log_file = os.path.join(self.output_base_dir, 'reward_metrics.jsonl')
            self.tool_usage_log_file = os.path.join(self.output_base_dir, 'tool_usage.jsonl')
            # self.debug_log_file = os.path.join(self.output_base_dir, 'debug_completions.log')
            
            os.makedirs(self.output_base_dir, exist_ok=True)
            logger.info(f"Logs will be saved to: {self.output_base_dir}")

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        return interArea / unionArea if unionArea > 0 else 0.0

    def _compute_hungarian_iou(self, pred_bboxes: List[List[float]], gt_bboxes: List[List[float]]) -> float:
        if not gt_bboxes: return 0.0
        if not pred_bboxes: return 0.0
        num_pred = len(pred_bboxes)
        num_gt = len(gt_bboxes)
        iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)
        for i, pred_box in enumerate(pred_bboxes):
            for j, gt_box in enumerate(gt_bboxes):
                iou_matrix[i, j] = self._calculate_iou(pred_box, gt_box)
        cost_matrix = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_ious = [float(iou_matrix[r, c]) for r, c in zip(row_indices, col_indices)]
        return float(np.mean(matched_ious)) if matched_ious else 0.0

    def _parse_bboxes_from_answer(self, text: str) -> List[List[float]]:
        pattern = re.compile(r'\((\d+)\s*,\s*(-?\d+)\)\s*,\s*\((\d+)\s*,\s*(-?\d+)\)')
        matches = pattern.findall(text)
        bboxes = []
        IMG_WIDTH = 832.0
        IMG_HEIGHT = 832.0
        
        for match in matches:
            try:
                x1_norm, y1_norm, x2_norm, y2_norm = map(int, match)
                x1 = x1_norm * IMG_WIDTH / 1000.0
                y1 = y1_norm * IMG_HEIGHT / 1000.0
                x2 = x2_norm * IMG_WIDTH / 1000.0
                y2 = y2_norm * IMG_HEIGHT / 1000.0
                bboxes.append([
                    min(x1, x2), min(y1, y2), 
                    max(x1, x2), max(y1, y2)
                ])
            except (ValueError, IndexError):
                continue
        return bboxes

    # ================= 绘图核心逻辑 (已添加 Step 标签) =================
    
    def _draw_plots(self):
        if not os.path.exists(self.metrics_log_file): return
        
        step_data = defaultdict(lambda: defaultdict(list))
        metrics_keys = ['avg_total', 'tamp_avg_total', 'tamp_avg_cls', 'tamp_avg_loc', 'tamp_avg_tool', 'real_avg_total', 'real_avg_cls', 'real_avg_loc', 'real_avg_tool']

        try:
            with open(self.metrics_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        step = item['step']
                        for key in metrics_keys:
                            if key in item: step_data[step][key].append(item.get(key, 0.0))
                    except: continue
        except: return

        if not step_data: return
        sorted_steps = sorted(step_data.keys())
        plot_data = {key: [] for key in metrics_keys}
        for step in sorted_steps:
            for key in metrics_keys:
                vals = step_data[step][key]
                avg = sum(vals) / len(vals) if vals else 0.0
                plot_data[key].append(avg)

        def smooth(scalars, weight=0.9):
            if not scalars: return []
            last = scalars[0]; smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed
        
        def format_axis(ax):
            # 设置主刻度为50，副刻度为10 (满足 grid every 10 steps 需求)
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            # 绘制网格：Minor 网格为 10 steps 的竖线
            ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
            ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.4) 
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=45)

        # Plot 1: Global Overview
        try:
            plt.figure(figsize=(12, 6))
            raw = plot_data['avg_total']
            sm = smooth(raw, self.smooth_weight)
            plt.plot(sorted_steps, raw, color='black', alpha=0.25)
            plt.plot(sorted_steps, sm, color='black', label='Global Avg', linestyle='--')
            
            raw_t = plot_data['tamp_avg_total']
            sm_t = smooth(raw_t, self.smooth_weight)
            plt.plot(sorted_steps, raw_t, color='#d62728', alpha=0.25)
            plt.plot(sorted_steps, sm_t, color='#d62728', label='Tampered Total')
            
            raw_r = plot_data['real_avg_total']
            sm_r = smooth(raw_r, self.smooth_weight)
            plt.plot(sorted_steps, raw_r, color='#2ca02c', alpha=0.25)
            plt.plot(sorted_steps, sm_r, color='#2ca02c', label='Real Total')
            
            plt.title(f'Training Reward Overview (w={self.smooth_weight})')
            plt.legend()
            
            # === 添加 Step 标签 ===
            plt.xlabel("Step") 
            format_axis(plt.gca())
            
            plt.savefig(os.path.join(self.output_base_dir, 'plot_1_global_overview_smooth.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception: pass

        # Plot 2: Detailed Split
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
            ax=axes[0,0]; d=plot_data['tamp_avg_cls']; ax.plot(sorted_steps, d, alpha=0.2); ax.plot(sorted_steps, smooth(d, self.smooth_weight), '#d62728', label='Tamp CLS'); ax.legend(); ax.set_title('Tamp CLS')
            ax=axes[0,1]; d=plot_data['real_avg_cls']; ax.plot(sorted_steps, d, alpha=0.2); ax.plot(sorted_steps, smooth(d, self.smooth_weight), '#2ca02c', label='Real CLS'); ax.legend(); ax.set_title('Real CLS')
            ax=axes[1,0]; d=plot_data['tamp_avg_loc']; ax.plot(sorted_steps, d, alpha=0.2); ax.plot(sorted_steps, smooth(d, self.smooth_weight), '#ff7f0e', label='Tamp LOC'); ax.legend(); ax.set_title('Tamp LOC')
            ax=axes[1,1]; d=plot_data['real_avg_loc']; ax.plot(sorted_steps, d, alpha=0.2); ax.plot(sorted_steps, smooth(d, self.smooth_weight), '#1f77b4', label='Real LOC'); ax.legend(); ax.set_title('Real LOC')
            ax=axes[2,0]; d=plot_data['tamp_avg_tool']; ax.plot(sorted_steps, d, alpha=0.2); ax.plot(sorted_steps, smooth(d, self.smooth_weight), '#9467bd', label='Tamp Tool'); ax.legend(); ax.set_title('Tamp Tool')
            ax=axes[2,1]; d=plot_data['real_avg_tool']; ax.plot(sorted_steps, d, alpha=0.2); ax.plot(sorted_steps, smooth(d, self.smooth_weight), '#17becf', label='Real Tool'); ax.legend(); ax.set_title('Real Tool')
            
            # === 遍历添加格式和底部的 Step 标签 ===
            for i, row in enumerate(axes):
                for ax in row: 
                    format_axis(ax)
                    if i == 2: # 仅在最后一行添加标签
                        ax.set_xlabel("Step")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_base_dir, 'plot_2_detailed_split_smooth.png'), dpi=150)
            plt.close()
        except Exception: pass

    def _draw_tool_plots(self):
        if not os.path.exists(self.tool_usage_log_file): return
        
        tool_keys = ['avg_ela', 'avg_fft', 'avg_npp', 'avg_zoom', 'tamp_ela', 'tamp_fft', 'tamp_npp', 'tamp_zoom', 'real_ela', 'real_fft', 'real_npp', 'real_zoom']
        step_data = defaultdict(lambda: defaultdict(list))
        try:
            with open(self.tool_usage_log_file, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        for k in tool_keys: 
                            if k in item: step_data[item['step']][k].append(item[k])
                    except: continue
        except: return

        if not step_data: return
        sorted_steps = sorted(step_data.keys())
        plot_data = {k: [] for k in tool_keys}
        for s in sorted_steps:
            for k in tool_keys:
                vals = step_data[s][k]
                plot_data[k].append(sum(vals)/len(vals) if vals else 0.0)

        def smooth(scalars, w=0.9):
            if not scalars: return []
            last = scalars[0]; res = []
            for p in scalars:
                val = last * w + (1-w)*p
                res.append(val); last = val
            return res
        
        def format_axis(ax):
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10)) # 保持每10步显示网格
            ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)
            ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.4)
            plt.setp(ax.get_xticklabels(), rotation=45)

        # Plot 3: Tool Usage Individual
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            colors = {'ela': '#d62728', 'fft': '#2ca02c', 'npp': '#1f77b4', 'zoom': '#9467bd'}
            def p_row(ax, prefix):
                for t in ['ela', 'fft', 'npp', 'zoom']:
                    k = f"{prefix}_{t}" if prefix != 'total' else f"avg_{t}"
                    d = plot_data[k]
                    ax.plot(sorted_steps, smooth(d, self.smooth_weight), color=colors[t], label=t.upper(), linewidth=2)
                ax.legend()
                format_axis(ax)
            p_row(axes[0], 'total'); axes[0].set_title('All Samples')
            p_row(axes[1], 'tamp'); axes[1].set_title('Tampered Samples')
            p_row(axes[2], 'real'); axes[2].set_title('Real Samples')
            
            # === 添加 Step 标签 ===
            axes[2].set_xlabel("Step")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_base_dir, 'plot_3_tool_usage_individual.png'), dpi=150)
            plt.close()
        except: pass

        # Plot 4: Tool Usage Grouped
        try:
            fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            def p_grp(ax, prefix):
                k_e = f"{prefix}_ela" if prefix != 'total' else f"avg_ela"
                k_f = f"{prefix}_fft" if prefix != 'total' else f"avg_fft"
                k_n = f"{prefix}_npp" if prefix != 'total' else f"avg_npp"
                k_z = f"{prefix}_zoom" if prefix != 'total' else f"avg_zoom"
                forensics = [plot_data[k_e][i]+plot_data[k_f][i]+plot_data[k_n][i] for i in range(len(plot_data[k_e]))]
                zoom = plot_data[k_z]
                ax.plot(sorted_steps, smooth(forensics, self.smooth_weight), color='#ff7f0e', label='Forensics', linewidth=2.5)
                ax.plot(sorted_steps, smooth(zoom, self.smooth_weight), color='#9467bd', label='Zoom', linewidth=2.5)
                ax.legend()
                format_axis(ax)
            p_grp(axes[0], 'total'); axes[0].set_title('All Samples')
            p_grp(axes[1], 'tamp'); axes[1].set_title('Tampered')
            p_grp(axes[2], 'real'); axes[2].set_title('Real')
            
            # === 添加 Step 标签 ===
            axes[2].set_xlabel("Step")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_base_dir, 'plot_4_tool_usage_grouped.png'), dpi=150)
            plt.close()
        except: pass

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        self.call_count += 1
        batch_total_rewards = []
        batch_logs = []
        
        is_tampered_list = kwargs.get('is_tampered', []) 
        bbox_objects_list = kwargs.get('bbox_objects', [])
        messages_list = kwargs.get('messages', [])

        batch_tool_stats = []

        for i in range(len(completions)):
            final_completion = completions[i]
            r_cls = 0.0
            r_loc = 0.0
            r_tool = 0.0
            iou_score = 0.0
            
            # --- 1. 工具统计 (修正：遍历整个对话历史) ---
            current_tool_usage = {'ela': 0, 'fft': 0, 'npp': 0, 'zoom': 0}
            
            # 定义一个内部函数来更新计数
            def update_usage_from_text(text_content):
                tc_json = parse_tool_call_from_text(text_content)
                if tc_json:
                    t_name = tc_json.get('name', '')
                    if t_name == 'ELA': current_tool_usage['ela'] = 1
                    elif t_name == 'FFT': current_tool_usage['fft'] = 1
                    elif t_name == 'NPP': current_tool_usage['npp'] = 1
                    elif t_name == 'zoom_in': current_tool_usage['zoom'] = 1
            
            # A. 遍历历史消息
            if i < len(messages_list):
                history = messages_list[i]
                for msg in history:
                    # 获取每条消息的内容
                    content = str(msg.get('content', ''))
                    # 尝试从中解析工具调用
                    update_usage_from_text(content)

            # B. 也要检查当前的 completion (防止工具调用正好发生在最后一步而没进 history)
            update_usage_from_text(final_completion)
            
            is_tampered = is_tampered_list[i] if i < len(is_tampered_list) else False
            batch_tool_stats.append({'is_tampered': is_tampered, 'usage': current_tool_usage})

            # === DEBUG LOGGING (Updated to show History) ===
            # if self.is_logging_process:
            #     try:
            #         with open(self.debug_log_file, 'a', encoding='utf-8') as df:
            #             log_str = f"\n=== Step {self.call_count} | BatchIdx {i} ===\n"
            #             # 记录历史中找到的工具
            #             log_str += f"[Detected Usage]: {json.dumps(current_tool_usage)}\n"
            #             # 简单的预览
            #             log_str += f"[Final Completion]: {final_completion[:100]} ...\n"
            #             df.write(log_str)
            #     except Exception as e:
            #         print(f"Debug Log Error: {e}")
            # # =====================
            
            # --- 2. 奖励计算逻辑 ---
            answer_content = extract_tag_content('answer', final_completion)
            
            # 这里的 tool_used_in_history 可以直接复用上面统计的结果
            # 只要任意一个工具被使用了，就算 used
            tool_used_in_history = any(v > 0 for v in current_tool_usage.values())
            
            gt_bboxes = []
            if is_tampered and i < len(bbox_objects_list) and isinstance(bbox_objects_list[i], dict):
                gt_bboxes = bbox_objects_list[i].get('bbox', [])

            if answer_content:
                ans_lower = answer_content.lower()
                parsed_bboxes = self._parse_bboxes_from_answer(answer_content)
                has_valid_bboxes = len(parsed_bboxes) > 0
                pred_real = ('real' in ans_lower) or ('authentic' in ans_lower)
                pred_fake = ('fake' in ans_lower) and has_valid_bboxes

                if is_tampered:
                    if pred_fake: r_cls = 1.0
                else:
                    if pred_real: r_cls = 1.0
                
                if is_tampered and gt_bboxes and has_valid_bboxes:
                    iou_score = self._compute_hungarian_iou(parsed_bboxes, gt_bboxes)
                    r_loc = iou_score * 2.0

                if tool_used_in_history:
                    if not is_tampered:
                        if r_cls == 1.0: r_tool = 0.5
                    else:
                        if r_cls == 1.0 and iou_score > 0.5: r_tool = 0.5

            total_reward = r_cls + r_loc + r_tool
            batch_total_rewards.append(total_reward)

            if self.is_logging_process:
                batch_logs.append({
                    "r_cls": r_cls, "r_loc": r_loc, "r_tool": r_tool,
                    "total": total_reward, "is_tampered": is_tampered, "iou": iou_score
                })

        # --- 3. 统一写日志与绘图 ---
        if self.is_logging_process:
            if batch_logs:
                avg_tot = float(np.mean([x['total'] for x in batch_logs]))
                tamp = [x for x in batch_logs if x['is_tampered']]
                real = [x for x in batch_logs if not x['is_tampered']]
                
                def sm(items, k): return float(np.mean([x[k] for x in items])) if items else 0.0

                l = {
                    "step": self.call_count,
                    "avg_total": avg_tot,
                    "tamp_cnt": len(tamp), "tamp_avg_total": sm(tamp, 'total'), "tamp_avg_cls": sm(tamp, 'r_cls'), "tamp_avg_loc": sm(tamp, 'r_loc'), "tamp_avg_tool": sm(tamp, 'r_tool'),
                    "real_cnt": len(real), "real_avg_total": sm(real, 'total'), "real_avg_cls": sm(real, 'r_cls'), "real_avg_loc": sm(real, 'r_loc'), "real_avg_tool": sm(real, 'r_tool')
                }
                with open(self.metrics_log_file, 'a') as f: f.write(json.dumps(l)+'\n')
            
            if batch_tool_stats:
                all_t = [x['usage'] for x in batch_tool_stats]
                tamp_t = [x['usage'] for x in batch_tool_stats if x['is_tampered']]
                real_t = [x['usage'] for x in batch_tool_stats if not x['is_tampered']]
                
                def cm(tl):
                    if not tl: return 0.0,0.0,0.0,0.0
                    return float(np.mean([t['ela'] for t in tl])), float(np.mean([t['fft'] for t in tl])), float(np.mean([t['npp'] for t in tl])), float(np.mean([t['zoom'] for t in tl]))

                ae, af, an, az = cm(all_t)
                te, tf, tn, tz = cm(tamp_t)
                re_v, rf, rn, rz = cm(real_t)
                
                tl = {
                    "step": self.call_count,
                    "avg_ela": ae, "avg_fft": af, "avg_npp": an, "avg_zoom": az,
                    "tamp_ela": te, "tamp_fft": tf, "tamp_npp": tn, "tamp_zoom": tz,
                    "real_ela": re_v, "real_fft": rf, "real_npp": rn, "real_zoom": rz
                }
                with open(self.tool_usage_log_file, 'a') as f: f.write(json.dumps(tl)+'\n')

            if self.call_count % self.plot_interval == 0:
                self._draw_plots()
                self._draw_tool_plots()
                gc.collect()

        return batch_total_rewards

# ================= 4. 多轮对话调度器 =================

class ForgeryDetectionScheduler(MultiTurnScheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self._original_image_paths = {}

    def _get_image_path_helper(self, infer_request, tool_name):
        request_id = getattr(infer_request, 'uuid', None) or id(infer_request)
        original_path = self._original_image_paths.get(request_id)
        if not original_path: original_path = get_original_image_path(infer_request)
        if not original_path and hasattr(infer_request, 'data_dict') and infer_request.data_dict:
            di = infer_request.data_dict.get('images', [])
            if di:
                f = di[0] if isinstance(di, list) else di
                if isinstance(f, str) and os.path.exists(f): original_path = f
        if not original_path: return None, None
        self._original_image_paths[request_id] = original_path
        
        path = ""
        if tool_name != 'zoom_in': path = get_tool_image_path(original_path, tool_name)
        return original_path, path

    def _sanitize_bbox(self, bbox: List[int], img_w: int, img_h: int) -> Tuple[List[int], bool]:
        MIN_CROP_SIZE = 32
        MAX_CROP_ASPECT_RATIO = 10.0 
        def _fb(): return [0, 0, max(1, img_w), max(1, img_h)]
        if not isinstance(bbox, list) or len(bbox) != 4: return _fb(), True
        try: x1, y1, x2, y2 = map(float, bbox)
        except: return _fb(), True
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
        w, h = x2-x1, y2-y1
        if w<1 or h<1: return _fb(), True
        ar = w/h
        if ar > MAX_CROP_ASPECT_RATIO:
            th = w/MAX_CROP_ASPECT_RATIO; d=(th-h)/2.0; y1-=d; y2+=d
        elif ar < 1.0/MAX_CROP_ASPECT_RATIO:
            tw = h/MAX_CROP_ASPECT_RATIO; d=(tw-w)/2.0; x1-=d; x2+=d
        res = [max(0, min(int(x1), img_w)), max(0, min(int(y1), img_h)), max(0, min(int(x2), img_w)), max(0, min(int(y2), img_h))]
        if res[2]-res[0]<=0 or res[3]-res[1]<=0: return _fb(), True
        return res, False

    def check_finished(self, infer_request: Any, response_choice: Any, current_turn: int) -> bool:
        content = response_choice.message.content
        if '<answer>' in content and '</answer>' in content: return True
        if self.max_turns and current_turn >= self.max_turns: return True
        
        tc = parse_tool_call_from_text(content)
        if not tc: return True 
        try:
            tn = tc.get('name')
            if tn not in ['ELA', 'FFT', 'NPP', 'zoom_in']: return True
            op, tp = self._get_image_path_helper(infer_request, tn)
            if tn == 'zoom_in':
                if not tc.get('arguments', {}).get('bbox'): return True
                if not op: return True
            else:
                if not tp or not os.path.exists(tp): return True
        except: return True
        return False

    def step(self, infer_request: Any, response_choice: Any, current_turn: int) -> Dict[str, Any]:
        self.call_count += 1
        completion = response_choice.message.content
        tc = parse_tool_call_from_text(completion)
        tool_name = tc.get('name') if tc else None
        
        orig_path, target_tool_path = self._get_image_path_helper(infer_request, tool_name)
        tool_img = None

        if tool_name == 'zoom_in':
            bbox = tc.get('arguments', {}).get('bbox')
            try:
                img = load_pil_image(orig_path)
                if img:
                    img = img.convert('RGB')
                    w, h = img.size
                    fb, _ = self._sanitize_bbox(bbox, w, h)
                    tool_img = img.crop(tuple(fb))
                    # Resize logic omitted for brevity, keeping original resizing logic is fine
                    # (Standardize size if needed)
                    # For completeness copying resize logic:
                    MIN_S, MAX_S, MAX_AR = 64, 1536, 8.0
                    tw, th = tool_img.size
                    if tw < MIN_S or th < MIN_S:
                        scale = max(MIN_S/tw, MIN_S/th); tool_img = tool_img.resize((int(tw*scale), int(th*scale)), Image.Resampling.BICUBIC); tw, th = tool_img.size
                    if tw > MAX_S or th > MAX_S:
                        scale = min(MAX_S/tw, MAX_S/th); tool_img = tool_img.resize((int(tw*scale), int(th*scale)), Image.Resampling.BICUBIC); tw, th = tool_img.size
                    ar = tw/th
                    if ar > MAX_AR: tool_img = tool_img.resize((tw, int(tw/MAX_AR)), Image.Resampling.NEAREST)
                    elif ar < 1.0/MAX_AR: tool_img = tool_img.resize((int(th/MAX_AR), th), Image.Resampling.NEAREST)
            except Exception as e:
                logger.error(f"Zoom fail: {e}")
        elif target_tool_path:
            # Decode the tool map before appending it to the multi-turn image
            # list. This preserves the previously validated rollout behavior.
            tool_img = load_pil_image(target_tool_path)
        
        query = "<tool_response>{\"status\": \"success\", \"tool_image\": \"<image>\"}</tool_response>"
        infer_request.messages.append({'role': 'user', 'content': query})
        if tool_img: infer_request.images.append(tool_img)
        if self.call_count % 5 == 0: gc.collect()
        return {'infer_request': infer_request, 'rollout_infos': {'images': infer_request.images}}

# ================= 5. 注册插件 =================
orms['forgery_detection_reward'] = ForgeryDetectionReward
multi_turns['forgery_detection_scheduler'] = ForgeryDetectionScheduler

print("Forgery Detection Plugin Loaded (With Step Labels).")
