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

import re, json
from typing import List, Union, Any
from .agent import *


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

# =========================
# 仿射传输奖励的辅助函数
# =========================

def _bbox_to_cxcywh(b: List[float]) -> tuple[float, float, float, float]:
    """[x1,y1,x2,y2] -> (cx, cy, w, h)，并确保 w,h 非负。"""
    x1, y1, x2, y2 = map(float, b)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy, w, h


def _transport_similarity(pred: List[float],
                          gt: List[float],
                          *,
                          alpha: float = 1.0,
                          beta: float = 1.0,
                          gamma: float = 0.0,
                          eps: float = 1e-6) -> float:
    """
    基于“把 B_p 通过仿射 T 搬运到 B_t”的思想定义连续相似度：
      - 线性部分 A = diag(sx, sy)，其中 sx = wt/wp, sy = ht/hp
      - 平移向量 t = (tx, ty) = (cx_t - sx*cx_p, cy_t - sy*cy_p)
      - 平移代价用  ||t|| / (r_p + r_t)  归一化，r 为外接圆半径
      - 尺度代价用 ||log(A)||_F = sqrt( (log sx)^2 + (log sy)^2 )
      - 可选纵横比代价 |log sx - log sy|

    能量：E = α*d_norm^2 + β*||log A||_F^2 + γ*Δ_ar^2
    相似度：R = exp(-E)，范围 (0,1]
    """
    cxp, cyp, wp, hp = _bbox_to_cxcywh(pred)
    cxt, cyt, wt, ht = _bbox_to_cxcywh(gt)

    # 避免 0 尺度导致除零 / log(0)
    wp = max(wp, eps)
    hp = max(hp, eps)
    wt = max(wt, eps)
    ht = max(ht, eps)

    # 线性缩放（各向异性）
    sx, sy = wt / wp, ht / hp

    # 仿射平移（按原点缩放后再平移）
    tx = cxt - sx * cxp
    ty = cyt - sy * cyp

    # 用外接圆半径和进行尺度无关的平移归一化
    rp = 0.5 * math.hypot(wp, hp)
    rt = 0.5 * math.hypot(wt, ht)
    d_norm = math.hypot(tx, ty) / max(rp + rt, eps)

    # 尺度/形状项：放大与缩小对称（log 惩罚）
    log_sx = math.log(max(sx, eps))
    log_sy = math.log(max(sy, eps))
    size_cost_sq = log_sx * log_sx + log_sy * log_sy

    # 可选：纵横比差（同样使用 log 对称）
    ar_cost_sq = (log_sx - log_sy) * (log_sx - log_sy)

    # 组合能量与相似度
    E = alpha * (d_norm ** 2) + beta * size_cost_sq + gamma * ar_cost_sq
    # 数值安全：避免 inf/NaN 造成崩溃
    if not math.isfinite(E):
        return 0.0
    R = math.exp(-E)
    # clamp 到 (0,1]
    if R < 0.0:
        R = 0.0
    elif R > 1.0:
        R = 1.0
    return R


def visual_reward(response: str, target_boxes: list) -> float:
    """
    基于“仿射传输 T”的双侧视觉奖励：
      R = 0.5 * (Recall + Precision)
    其中单对相似度 sim(pred, gt) 由 _transport_similarity 定义，兼顾中心位移与尺度/形状差异，
    并对所有距离提供非零、平滑的梯度信号。

    - response: 模型输出字符串，包含 <box>[x1,y1,x2,y2]</box> 标签
    - target_boxes: GT 框列表 [[x1,y1,x2,y2], ...]
    返回 [0,1] 的标量奖励
    """
    # ---- 1) 解析预测框 ----
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

    # ---- 2) 清洗 GT 框 ----
    gt_boxes: List[List[float]] = []
    for b in (target_boxes or []):
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
            if x1 < x2 and y1 < y2:
                gt_boxes.append([x1, y1, x2, y2])

    N = len(pred_boxes)
    M = len(gt_boxes)

    # 角落情形：无预测或无 GT，返回 0（与原逻辑一致）
    if N == 0 or M == 0:
        return 0.0

    # ---- 3) Recall：对每个 GT，取与任一 Pred 的最大传输相似度 ----
    recall_terms: List[float] = []
    for gt in gt_boxes:
        best = 0.0
        for pred in pred_boxes:
            best = max(best, _transport_similarity(pred, gt))
        recall_terms.append(best)
    recall_sim = sum(recall_terms) / float(M) if M > 0 else 0.0

    # ---- 4) Precision：对每个 Pred，取与任一 GT 的最大传输相似度 ----
    precision_terms: List[float] = []
    for pred in pred_boxes:
        best = 0.0
        for gt in gt_boxes:
            best = max(best, _transport_similarity(pred, gt))
        precision_terms.append(best)
    precision_sim = sum(precision_terms) / float(N) if N > 0 else 0.0

    # ---- 5) 双侧奖励 ----
    reward = 0.5 * (recall_sim + precision_sim)

    # 数值安全
    if reward < 0.0:
        reward = 0.0
    elif reward > 1.0:
        reward = 1.0

    return reward


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1, visual_weight: float = 0.1) -> list[dict[str, float]]:
    '''
     visual_weight是视觉奖励的超参数，论文中没有给，感觉得调参了
    '''
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        # 解析目标框
        target_boxes = parse_bboxes(reward_input["target_instances"])
        # 使用仿射传输相似度版本
        visual_score = visual_reward(response, target_boxes)

        scores.append(
            {
                "overall": (1 - format_weight - visual_weight) * accuracy_score + format_weight * format_score + visual_weight * visual_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": visual_score,
            }
        )

    return scores