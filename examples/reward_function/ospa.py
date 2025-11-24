# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re, json, sys, os, math
import numpy as np
from typing import List, Union, Any
from itertools import permutations
from scipy.optimize import linear_sum_assignment

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from agent import (
    _extract_ans,
    _looks_like_mc,
    _tokenize_mc,
    _judge_with_llm,
)

def format_reward(response: str) -> float:
    # 满足：先有 <think>…</think>，随后出现 <answer>…</answer> 或 \boxed{…}
    pattern = re.compile(
        r"(?s)<think>.*?</think>\s*(?:<answer>.*?</answer>|\\boxed\{.*?\})"
    )
    return 1.0 if pattern.search(response) else 0.0

    
def accuracy_reward(response: str, ground_truth: str, *, question: str = "") -> float:
    """
    Utilize exact-matching for multiple-choice questions, and leverage an online
    reward model (Qwen2.5-72B-Instruct) to judge whether the prediction is correct
    given the question and the ground-truth answer.

    返回：1.0（正确）或 0.0（错误）
    """
    ans = _extract_ans(response)
    if not ans:
        return 0.0

    # 1) 选择题：精确匹配（多选无序匹配）
    if _looks_like_mc(ans, ground_truth):
        return 1.0 if set(_tokenize_mc(ans)) == set(_tokenize_mc(ground_truth)) else 0.0

    # 2) 非选择题：LLM-as-a-judge（给出 1/0）
    return _judge_with_llm(ans, ground_truth.strip(), question=question)


def _extract_from_list(objs: List[Any]) -> List[List[float]]:
    out: List[List[float]] = []
    for obj in objs:
        if isinstance(obj, dict) and "bbox" in obj:
            bbox = obj["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # 强制转成 float，防止出现 int/str 混合
                x1, y1, x2, y2 = bbox
                out.append([float(x1), float(y1), float(x2), float(y2)])
    return out

    
def parse_bboxes(value: Union[str, list]) -> List[List[float]]:
    """
    把 CSV 单元格中的 target_instances 字段解析成 [ [x1,y1,x2,y2], ... ]。
    - value 可以是 JSON 字符串，或已经是 list（比如使用 JSON/JSONL 数据集时）。
    - 若字符串中包含多个 JSON 数组，逐个解析并合并。
    """
    # 1) 已经是结构化对象（如 list）：
    if isinstance(value, list):
        return _extract_from_list(value)

    # 2) 字符串：先尝试整体解析为一个数组
    s = (value or "").strip()
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return _extract_from_list(data)
    except json.JSONDecodeError:
        pass

    # 3) 兜底：字符串里可能包含多个数组，逐段提取再解析
    bboxes: List[List[float]] = []
    # 这个正则粗略匹配最外层的 [ {...}, {...}, ... ] 片段
    for m in re.finditer(r'\[\s*(?:\{.*?\})\s*(?:,\s*\{.*?\}\s*)*\]', s, flags=re.S):
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                bboxes.extend(_extract_from_list(arr))
        except json.JSONDecodeError:
            continue
    return bboxes


# ========= OSPA 子模式匹配：几何距离（中心+面积比） =========
def _box_geom_distance(b1, b2, alpha: float = 0.6, eps: float = 1e-9) -> float:
    """
    d ∈ [0, ~1.5] 左右的连续几何距离：
      d = alpha * |c1-c2|/(r1+r2) + (1-alpha) * (1 - min(A1, A2)/max(A1, A2))
    """
    x1, y1, x2, y2 = map(float, b1)
    u1, v1, u2, v2 = map(float, b2)

    w1, h1 = max(eps, x2 - x1), max(eps, y2 - y1)
    w2, h2 = max(eps, u2 - u1), max(eps, v2 - v1)

    c1x, c1y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    c2x, c2y = 0.5 * (u1 + u2), 0.5 * (v1 + v2)

    r1 = 0.5 * math.hypot(w1, h1)
    r2 = 0.5 * math.hypot(w2, h2)

    # 外接圆心间距（尺度无关）
    d_center = math.hypot(c1x - c2x, c1y - c2y) / (r1 + r2 + eps)

    # 面积比差异（0 表示相同面积，1 表示极不相称）
    a1, a2 = w1 * h1, w2 * h2
    d_area = 1.0 - (min(a1, a2) / (max(a1, a2) + eps))

    return float(alpha * d_center + (1.0 - alpha) * d_area)


# ========= OSPA 主体 + 软门控数量惩罚 =========
def _ospa_components(pred_boxes: list, gt_boxes: list,
                     *, c: float = 0.8, p: int = 1, alpha: float = 0.6):
    """
    计算 OSPA 的两部分组成（几何 / 数量）：
      - 构造 k×k 方阵代价：D_ij = min(c, d(bi, bj))^p；dummy 的代价 = c^p
      - 求最小匹配
      - 返回：
          geom_term_norm ∈ [0,1]   （匹配几何成本，经 OSPA 归一化并除以 c）
          cnt_term_norm  ∈ [0,1]   （未匹配数量成本，经 OSPA 归一化并除以 c）
          delta_n = |#pred - #gt|
          pairs = [(i,j), ...]     （真实匹配对）
    """
    m, n = len(pred_boxes), len(gt_boxes)
    k = max(m, n)
    if k == 0:
        return 0.0, 0.0, 0, []

    C = np.full((k, k), fill_value=(c ** p), dtype=float)
    for i in range(m):
        for j in range(n):
            d = _box_geom_distance(pred_boxes[i], gt_boxes[j], alpha=alpha)
            C[i, j] = min(c, d) ** p

    rows, cols = linear_sum_assignment(C)
    # 真实匹配对（非 dummy）
    pairs = [(i, j) for i, j in zip(rows, cols) if i < m and j < n and C[i, j] < (c ** p - 1e-9)]

    # OSPA 按 k 归一化
    total_cost_p = C[rows, cols].sum()          # = geom_cost_p + unmatched_count * c^p
    unmatched_count = k - len(pairs)
    geom_cost_p = total_cost_p - unmatched_count * (c ** p)

    # p-范数复原，再除以 c 映到 [0,1]
    geom_term = ((geom_cost_p / k) ** (1.0 / p)) if geom_cost_p > 0 else 0.0
    cnt_term  = ((unmatched_count * (c ** p) / k) ** (1.0 / p)) if unmatched_count > 0 else 0.0

    geom_term_norm = geom_term / c if c > 1e-9 else 0.0
    cnt_term_norm  = cnt_term  / c if c > 1e-9 else 0.0

    delta_n = abs(m - n)
    return float(geom_term_norm), float(cnt_term_norm), int(delta_n), pairs


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def visual_reward_ospa(response: str, target_boxes: list,
                       *,
                       # OSPA 参数
                       c: float = 0.8, p: int = 1, alpha: float = 0.6,
                       # 软门控：小量差几何主导；大量差强惩罚
                       d0: float = 2.0,      # 几何主导阈值中心（Δn≈2 以内）
                       d1: float = 3.5,      # 数量主导阈值中心（Δn≥3~4）
                       sharp: float = 0.5,   # 门控陡峭度
                       lam_cnt: float = 2.0, # 数量惩罚放大系数
                       eta: float = 2.0      # 指数映射陡峭度
                       ) -> float:
    """
    OSPA + 软门控数量惩罚的视觉奖励 ∈ (0,1]：
      1) 用 OSPA 做子模式匹配，分解出几何项/数量项两个“归一化成本”
      2) 根据 Δn 用 sigmoid 做软门控：Δn≤2 → 几何主导；Δn≥3~4 → 数量主导
      3) cost_all = w_geom*geom + lam_cnt*w_cnt*cnt
      4) reward = exp(-eta*cost_all)
    """
    # 解析预测框（沿用你原来的浮点解析）
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, response, re.DOTALL)

    pred_boxes: List[List[float]] = []
    for match in matches:
        box = match.strip()
        coord_pattern = r'\[(\-?\d+\.?\d*),(\-?\d+\.?\d*),(\-?\d+\.?\d*),(\-?\d+\.?\d*)\]'
        coord_match = re.match(coord_pattern, box)
        if coord_match:
            x1, y1, x2, y2 = map(float, coord_match.groups())
            if x1 < x2 and y1 < y2:
                pred_boxes.append([x1, y1, x2, y2])

    gt_boxes: List[List[float]] = []
    for b in (target_boxes or []):
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
            if x1 < x2 and y1 < y2:
                gt_boxes.append([x1, y1, x2, y2])

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    geom_term, cnt_term, delta_n, _pairs = _ospa_components(
        pred_boxes, gt_boxes, c=c, p=p, alpha=alpha
    )

    # 软门控权重
    w_geom = 1.0 - _sigmoid((delta_n - d0) / sharp)
    w_cnt  = _sigmoid((delta_n - d1) / sharp)

    cost_all = w_geom * geom_term + lam_cnt * w_cnt * cnt_term
    cost_all = min(1.0, max(0.0, float(cost_all)))
    reward = math.exp(-eta * cost_all)
    reward = min(1.0, max(0.0, float(reward)))
    return reward


def compute_box_iou(predict_str: str, target_boxes: list) -> float:
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, predict_str, re.DOTALL)

    all_boxes = []
    
    for match in matches:
        box = match.strip()
        
        coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
        coord_match = re.match(coord_pattern, box)
        
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            
            if x1 < x2 and y1 < y2:
                # all_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                all_boxes.append([x1, y1, x2, y2])

    def calculate_average_iou(pred_boxes, target_boxes):
        def compute_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_width = max(0, inter_x_max - inter_x_min)
            inter_height = max(0, inter_y_max - inter_y_min)
            inter_area = inter_width * inter_height

            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)

            union_area = area1 + area2 - inter_area

            return inter_area / union_area if union_area > 0 else 0.0

        pred_coords = pred_boxes
        target_coords = target_boxes # x1,y1,x2,y2

        total_iou = 0.0
        num_targets = len(target_boxes)

        if num_targets == 0:
            return 0.0

        for t_coord in target_coords:
            best_iou = 0.0
            for p_coord in pred_coords:
                iou = compute_iou(t_coord, p_coord)
                if iou > best_iou:
                    best_iou = iou
            total_iou += best_iou

        return total_iou / num_targets

    return calculate_average_iou(all_boxes, target_boxes)


def compute_score(reward_inputs: list[dict[str, Any]], accuracy_weight: float = 0.9, format_weight: float = 0.1, visual_weight: float = 1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []  
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"], question=reward_input["question"])
        ### parsing target bbox
        target_boxes = parse_bboxes(reward_input["target_instances"]) 
        visual_score = visual_reward_ospa(response, target_boxes) 
        mIoU = compute_box_iou(response, target_boxes) 
        
        scores.append(
            {
                "overall":  accuracy_weight*accuracy_score + format_weight*format_score + visual_weight*accuracy_score*visual_score, 
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": visual_score,
                "mIoU": mIoU,
            }
        )

    return scores
