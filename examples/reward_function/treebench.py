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

import sys,os
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


def visual_reward(response: str, target_boxes: list) -> float:
    """
    Compute the dual-IoU visual reward exactly as in TreeVGR §4.2:
      R_IoU = 0.5 * (Recall_IoU + Precision_IoU)
    where:
      Recall_IoU   = (1/M) * sum_k max_i IoU(pred_i, gt_k)
      Precision_IoU= (1/N) * sum_i max_k IoU(gt_k, pred_i)

    - response: model output string containing <box>[x1,y1,x2,y2]</box> tags
    - target_boxes: list of GT boxes [[x1,y1,x2,y2], ...]

    Returns:
      reward in [0, 1]
    """
    # ---- 1) Parse predicted boxes from <box>...</box> ----
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, response, re.DOTALL)

    pred_boxes = []
    for match in matches:
        box = match.strip()
        coord_pattern = r'\[(\-?\d+\.?\d*),(\-?\d+\.?\d*),(\-?\d+\.?\d*),(\-?\d+\.?\d*)\]'
        coord_match = re.match(coord_pattern, box)
        if coord_match:
            x1, y1, x2, y2 = map(float, coord_match.groups())
            if x1 < x2 and y1 < y2:
                pred_boxes.append([x1, y1, x2, y2])

    # ---- 2) Sanitize GT boxes ----
    gt_boxes = []
    for b in (target_boxes or []):
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = map(float, b)
            if x1 < x2 and y1 < y2:
                gt_boxes.append([x1, y1, x2, y2])

    N = len(pred_boxes)
    M = len(gt_boxes)

    # Corner cases: if there are no preds or no GTs, return 0 reward (conservative).
    if N == 0 or M == 0:
        return 0.0

    # ---- 3) IoU helper for [x1,y1,x2,y2] ----
    def iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    # ---- 4) Recall term: average over GTs of best IoU with any prediction ----
    recall_terms = []
    for gt in gt_boxes:
        best = 0.0
        for pred in pred_boxes:
            best = max(best, iou_xyxy(pred, gt))
        recall_terms.append(best)
    recall_iou = sum(recall_terms) / float(M) if M > 0 else 0.0

    # ---- 5) Precision term: average over predictions of best IoU with any GT ----
    precision_terms = []
    for pred in pred_boxes:
        best = 0.0
        for gt in gt_boxes:
            best = max(best, iou_xyxy(gt, pred))
        precision_terms.append(best)
    precision_iou = sum(precision_terms) / float(N) if N > 0 else 0.0

    # ---- 6) Dual IoU reward ----
    reward = 0.5 * (recall_iou + precision_iou)

    # Numerical safety
    if reward < 0.0:
        reward = 0.0
    elif reward > 1.0:
        reward = 1.0

    return reward

    
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
        visual_score = visual_reward(response, target_boxes) 
        
        scores.append(
            {
                "overall":  accuracy_weight*accuracy_score + format_weight*format_score + visual_weight*visual_score, 
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": visual_score,
            }
        )

    return scores
