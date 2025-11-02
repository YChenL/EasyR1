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

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


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
    pattern = r"<box>(.*?)</box>"
    matches = re.findall(pattern, response, re.DOTALL)

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
                
    '''
     all_boxes: 预测的box
     target_boxes: gt的box
    '''
    #reward的计算方式:
    # step 1: 匹配all_boxes中每个box与target_boxes中最接近的那个; ***特别地，由于为每个predict中的bbox计算reward，会奖励模型多预测box，但是我们应该惩罚无关的box避免模型一直输出无意义的box，如何做？
              # 论文中是用查准率来平衡reward, 我们看看能否设计一个新的方式
    # step 2: 为匹配上的box计算reward! 求预测box与target的外接圆，连接圆形，计算角度与距离，用欧拉公式合成reward标量
    return 

    
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
        ### parsing target bbox
        target_boxes = parse_bboxes(reward_input["target_instances"]) 
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
