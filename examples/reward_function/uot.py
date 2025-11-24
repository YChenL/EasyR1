# Copyright 2024 Bytedance ...
import re, json
from typing import List, Union, Any

import sys, os
import math
import numpy as np

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
    pattern = re.compile(r"(?s)<think>.*?</think>\s*(?:<answer>.*?</answer>|\\boxed\{.*?\})")
    return 1.0 if pattern.search(response) else 0.0

def accuracy_reward(response: str, ground_truth: str, *, question: str = "") -> float:
    ans = _extract_ans(response)
    if not ans:
        return 0.0
    if _looks_like_mc(ans, ground_truth):
        return 1.0 if set(_tokenize_mc(ans)) == set(_tokenize_mc(ground_truth)) else 0.0
    return _judge_with_llm(ans, ground_truth.strip(), question=question)

def _extract_from_list(objs: List[Any]) -> List[List[float]]:
    out: List[List[float]] = []
    for obj in objs:
        if isinstance(obj, dict) and "bbox" in obj:
            bbox = obj["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                out.append([float(x1), float(y1), float(x2), float(y2)])
    return out

def parse_bboxes(value: Union[str, list]) -> List[List[float]]:
    if isinstance(value, list):
        return _extract_from_list(value)
    s = (value or "").strip()
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return _extract_from_list(data)
    except json.JSONDecodeError:
        pass
    bboxes: List[List[float]] = []
    for m in re.finditer(r'\[\s*(?:\{.*?\})\s*(?:,\s*\{.*?\}\s*)*\]', s, flags=re.S):
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                bboxes.extend(_extract_from_list(arr))
        except json.JSONDecodeError:
            continue
    return bboxes

# =========================
# UOT：辅助函数
# =========================
def _bbox_to_cxcywh(b: List[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, b)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy, w, h

def _pairwise_geom_cost_raw(pred_boxes: List[List[float]],
                            gt_boxes: List[List[float]],
                            *,
                            alpha: float = 1.0,
                            beta: float = 1.0,
                            gamma: float = 0.0,
                            eps: float = 1e-8) -> np.ndarray:
    """
    构造“尺度无关”的几何代价（未归一化）：
      c = alpha * d_center^2 + beta * d_size^2 + gamma * d_ar^2
      d_center = ||c_p - c_g|| / (r_p + r_g)
      d_size   = sqrt( (log(w_g/w_p))^2 + (log(h_g/h_p))^2 )
      d_ar     = |log(w_g/w_p) - log(h_g/h_p)|
    """
    N, M = len(pred_boxes), len(gt_boxes)
    C = np.zeros((N, M), dtype=np.float64)
    for i, pb in enumerate(pred_boxes):
        cxp, cyp, wp, hp = _bbox_to_cxcywh(pb)
        wp = max(wp, eps); hp = max(hp, eps)
        rp = 0.5 * math.hypot(wp, hp)
        for j, gb in enumerate(gt_boxes):
            cxt, cyt, wt, ht = _bbox_to_cxcywh(gb)
            wt = max(wt, eps); ht = max(ht, eps)
            rt = 0.5 * math.hypot(wt, ht)
            d_center = math.hypot(cxp - cxt, cyp - cyt) / max(rp + rt, eps)
            log_sx = math.log(wt / wp); log_sy = math.log(ht / hp)
            d_size = math.sqrt(log_sx * log_sx + log_sy * log_sy)
            d_ar = abs(log_sx - log_sy)
            C[i, j] = alpha * (d_center ** 2) + beta * (d_size ** 2) + gamma * (d_ar ** 2)
    return C

def _normalize_cost_matrix(C_raw: np.ndarray, q: float = 0.95, tiny: float = 1e-12) -> np.ndarray:
    """
    将几何代价矩阵缩放到 [0,1]，避免分辨率/尺度偏置。
    用分位数缩放：C = clip(C_raw / quantile(C_raw, q), 0, 1)。
    对极端小矩阵或恒等值落在 tiny 时，退化到除以 (max-min) 的 min-max。
    """
    qval = float(np.quantile(C_raw, q)) if C_raw.size > 0 else 1.0
    if qval > tiny:
        C = C_raw / qval
        C = np.clip(C, 0.0, 1.0)
        return C
    # 退化：min-max
    cmin = float(np.min(C_raw)) if C_raw.size > 0 else 0.0
    cmax = float(np.max(C_raw)) if C_raw.size > 0 else 1.0
    denom = max(cmax - cmin, tiny)
    C = (C_raw - cmin) / denom
    C = np.clip(C, 0.0, 1.0)
    return C

def _gkl(p: np.ndarray, q: np.ndarray, tiny: float = 1e-12) -> float:
    """
    广义 KL：KL(p||q) = sum p log(p/q) - p + q ；支持 p 或 q 含零，数值安全。
    要求 p,q >= 0。
    """
    p = np.maximum(p, 0.0)
    q = np.maximum(q, tiny)
    term = p * (np.log(np.maximum(p, tiny)) - np.log(q)) - p + q
    val = float(np.sum(term))
    return max(val, 0.0)

def _uot_sinkhorn(C: np.ndarray,
                  a: np.ndarray,
                  b: np.ndarray,
                  *,
                  epsilon: float = 0.05,
                  rho_a: float = 1.0,
                  rho_b: float = 1.0,
                  max_iter: int = 200,
                  tol: float = 1e-9) -> tuple[float, float, float]:
    """
    KL-UOT 的广义 Sinkhorn。返回：
      avg_cost = <P, C> / sum(P)                （传输平均代价，C∈[0,1] ⇒ avg_cost∈[0,1]）
      kl_r     = KL(r || a)  源边缘偏差（消除质量）
      kl_c     = KL(c || b)  目标边缘偏差（创造质量）
    """
    tiny = 1e-12
    K = np.exp(-C / max(epsilon, tiny)).astype(np.float64) + tiny

    tau_a = rho_a / (rho_a + epsilon)
    tau_b = rho_b / (rho_b + epsilon)

    N, M = C.shape
    u = np.ones(N, dtype=np.float64)
    v = np.ones(M, dtype=np.float64)

    for _ in range(max_iter):
        u_prev, v_prev = u, v
        Kv = K @ v
        Kv = np.maximum(Kv, tiny)
        u = (a / Kv) ** tau_a

        KTu = K.T @ u
        KTu = np.maximum(KTu, tiny)
        v = (b / KTu) ** tau_b

        if (np.max(np.abs(np.log(u + tiny) - np.log(u_prev + tiny))) < tol and
            np.max(np.abs(np.log(v + tiny) - np.log(v_prev + tiny))) < tol):
            break

    P = (u[:, None] * K) * v[None, :]
    mass = float(np.sum(P))
    if mass <= tiny:
        # 极端退化：全部惩罚
        return 1.0, float('inf'), float('inf')

    avg_cost = float(np.sum(P * C) / mass)

    r = np.sum(P, axis=1)        # 源边缘（输送出的质量）
    c = np.sum(P, axis=0)        # 目标边缘（接收的质量）

    kl_r = _gkl(r, a)            # 消除质量代价
    kl_c = _gkl(c, b)            # 创造质量代价
    return avg_cost, kl_r, kl_c

# =========================
# UOT 版 visual_reward（C 与质量正则显式归一化 + 最终一次性归一化）
# =========================
def visual_reward(response: str,
                  target_boxes: list,
                  *,
                  # 几何代价参数
                  alpha: float = 1.0,
                  beta: float = 1.0,
                  gamma: float = 0.0,
                  c_norm_quantile: float = 0.95,
                  # UOT & Sinkhorn 超参
                  epsilon: float = 0.05,
                  rho_a: float = 1.0,
                  rho_b: float = 1.0,
                  max_iter: int = 200,
                  # 质量正则的归一化尺度（score = exp(-kl_scale * KL)）
                  kl_scale: float = 1.0,
                  # 最终一次性归一化温度
                  eta: float = 1.0) -> float:
    """
    1) 解析预测/GT 框；
    2) 几何代价 C_raw（尺度无关）→ 分位数归一化到 [0,1] 得 C；
    3) UOT（广义 Sinkhorn）得：
         - avg_cost ∈ [0,1]
         - kl_r = KL(r||a), kl_c = KL(c||b)
       将 kl_* 各自映射为 score_* = exp(-kl_scale * kl_*) ∈ (0,1]；
    4) 组合综合代价：
         cost_all = 0.5*avg_cost + 0.25*(1-score_c) + 0.25*(1-score_r)
    5) 最终一次性归一化：reward = exp(-eta * cost_all) ∈ (0,1]
    """
    # ---- 解析预测框 ----
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

    # ---- 清洗 GT 框 ----
    gt_boxes: List[List[float]] = []
    for b in (target_boxes or []):
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
            if x1 < x2 and y1 < y2:
                gt_boxes.append([x1, y1, x2, y2])

    N = len(pred_boxes)
    M = len(gt_boxes)
    if N == 0 or M == 0:
        return 0.0

    # ---- C_raw → 归一化到 [0,1] ----
    C_raw = _pairwise_geom_cost_raw(pred_boxes, gt_boxes, alpha=alpha, beta=beta, gamma=gamma)
    C = _normalize_cost_matrix(C_raw, q=c_norm_quantile)

    # ---- 源/汇质量（均匀；UOT 通过 rho_* 允许质量不守恒）----
    a = np.full(N, 1.0 / N, dtype=np.float64)
    b = np.full(M, 1.0 / M, dtype=np.float64)

    # ---- UOT：传输成本 + 质量正则（KL）----
    avg_cost, kl_r, kl_c = _uot_sinkhorn(C, a, b, epsilon=epsilon, rho_a=rho_a, rho_b=rho_b, max_iter=max_iter)

    if not math.isfinite(avg_cost):
        return 0.0

    # ---- 将 KL 映射为 [0,1] 的“质量一致性得分” ----
    score_create  = math.exp(-max(0.0, kl_scale) * max(0.0, kl_c))  # 目标边缘：创造质量一致性
    score_destroy = math.exp(-max(0.0, kl_scale) * max(0.0, kl_r))  # 源边缘：消除质量一致性

    # ---- 综合代价（越小越好）----
    # cost_all = 0.5 * avg_cost + 0.25 * (1.0 - score_create) + 0.25 * (1.0 - score_destroy)
    cost_all = 0.8 * avg_cost + 0.1 * (1.0 - score_create) + 0.1 * (1.0 - score_destroy)
    cost_all = max(0.0, min(1.0, cost_all))  # 安全夹取

    # ---- 最终一次性归一化 ----
    reward = math.exp(-max(0.0, eta) * cost_all)
    reward = 0.0 if reward < 0.0 else (1.0 if reward > 1.0 else reward)
    return reward

# =========================
# 其它维持不变的辅助
# =========================
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

        total_iou = 0.0
        num_targets = len(target_boxes)
        if num_targets == 0:
            return 0.0
        for t in target_boxes:
            best = 0.0
            for p in pred_boxes:
                best = max(best, compute_iou(t, p))
            total_iou += best
        return total_iou / num_targets

    return calculate_average_iou(all_boxes, target_boxes)

def compute_score(reward_inputs: list[dict[str, Any]], accuracy_weight: float = 0.9, format_weight: float = 0.1, visual_weight: float = 1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"], question=reward_input["question"])
        target_boxes = parse_bboxes(reward_input["target_instances"])
        visual_score = visual_reward(
            response, target_boxes,
            alpha=1.0, beta=1.0, gamma=0.0,
            c_norm_quantile=0.95,
            epsilon=0.05, rho_a=1.0, rho_b=1.0, max_iter=200,
            kl_scale=1.0,
            eta=1.0
        )
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
