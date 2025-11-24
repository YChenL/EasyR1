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

from utils import (
    _all_boxes_valid,
    _extract_ans,
    _normalize_yes_no,
    _extract_number_strict,
    _normalize_choice_token,
    _parse_mc_options,
    _read_raw_image,
    _build_message,
)
from openai import OpenAI

# vLLM OpenAI 兼容服务的地址和模型名
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vllm 默认不会校验
VLLM_RM_MODEL = os.getenv("VLLM_RM_MODEL", "Qwen3-VL-8B-Instruct")  # 改成你实际的 model 名称

vllm_client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

def format_reward(response: str) -> float:
    """
    满足以下条件给 1.0，否则 0.0：
    1. 整体结构为（按顺序且各出现一次）：
       <analysis>...</analysis>
       <localize>...</localize>
       <thinking>...</thinking>
       <answer>...</answer>
    2. <localize>...</localize> 中：
       - 至少包含一个 <box>...</box>
       - 且所有 box 都是严格的 [x1,y1,x2,y2]，并满足坐标约束。
    """
    pattern = re.compile(
        r"(?s)^\s*"
        r"<analysis>.*?</analysis>\s*"
        r"<localize>(?P<localize>.*?)</localize>\s*"
        r"<thinking>.*?</thinking>\s*"
        r"<answer>.*?</answer>\s*$"
    )

    m = pattern.search(response)
    if not m:
        return 0.0

    localize_block = m.group("localize")
    if not _all_boxes_valid(localize_block):
        return 0.0

    return 1.0


def accuracy_reward(response: str, ground_truth: str, *) -> float:
    """
    针对三类题型进行“可验证”的精确匹配：
      (1) 选择题（单选 / 多选）：
          - 形式如: "A", "A,B,C", "A B C", "(A), (B)", "(A) (C)" 等
          - 分隔符可以是逗号或空白
          - 选项不区分大小写
          - 选项可以被括号 () 包裹
          - 多选按集合匹配（无序）
      (2) 判断题（Yes/No）：
          - ground_truth 为 "yes" 或 "no"，不区分大小写
      (3) 数学题 / 计数题：
          - 答案仅包含连续数字，如 "3", "72", "1234"
          - 通过整数值精确匹配

    所有题型都不再调用 LLM-as-a-judge，只做规则匹配。
    答对返回 1.0，答错返回 0.0。
    """
    # 1. 提取 <answer>...</answer> 中的预测答案
    ans = _extract_ans(response)  # 假设你已有这个函数
    if not ans:
        return 0.0

    ans = ans.strip()
    gt = (ground_truth or "").strip()
    if not gt:
        # 没有 ground_truth，就视作无法验证（按 0 处理）
        return 0.0

    # ========= 先根据 ground_truth 判断题型 =========

    # (2) 判断题：Yes/No
    gt_yn = _normalize_yes_no(gt)
    if gt_yn is not None:
        ans_yn = _normalize_yes_no(ans)
        return 1.0 if ans_yn is not None and ans_yn == gt_yn else 0.0

    # (3) 数学题 / 计数题：纯数字
    gt_num = _extract_number_strict(gt)
    if gt_num is not None:
        ans_num = _extract_number_strict(ans)
        return 1.0 if ans_num is not None and ans_num == gt_num else 0.0

    # (1) 选择题（默认剩下的都当作选择题来处理）
    gt_mc = _parse_mc_options(gt)
    ans_mc = _parse_mc_options(ans)

    # ground_truth 解析失败（极少见，说明标注不符合规则）→ 直接按错误处理
    if not gt_mc:
        return 0.0

    # 模型答案解析失败，也判错
    if not ans_mc:
        return 0.0

    # 多选无序精确匹配
    return 1.0 if ans_mc == gt_mc else 0.0


def visual_reward(img: str, question: str, response: str, ground_truth: str) -> float:
    """
    使用 vLLM OpenAI 兼容 HTTP 服务的 Qwen-VL reward model 计算视觉感知 reward。

    步骤：
    1) 用 Round1 的输出 `response` 中的 <box> + <localize>，构造多图 interleaved messages。
    2) 调用 vLLM OpenAI-compat server，要求返回 token 级 logprobs。
    3) 用 accuracy_reward(rm_response, ground_truth) 判断 reward model 的答案是否正确。
       - 若错误：视觉 reward = 0.0
       - 若正确：视觉 reward = 生成 token 的平均 logprob
         （即 log P(Y | X, evidence) 的 token 平均值）
    """

    # 1) 构造多模态 messages（包含 base64 图片）
    messages = _build_message(img, question, response)

    try:
        # 2) 调 vLLM OpenAI 兼容接口，开启 logprobs
        completion = vllm_client.chat.completions.create(
            model=VLLM_RM_MODEL,
            messages=messages,
            temperature=0.01,
            top_p=0.001,
            max_tokens=1024,
            n=1,
            # ChatCompletion 标准参数
            logprobs=True,
            top_logprobs=1,
            # vLLM 特有采样参数
            extra_body={
                "top_k": 1,
                "repetition_penalty": 1.0,
            },
        )
    except Exception:
        # 远端 RM 调用失败时，保守返回 0，避免整个 RL 进程挂掉
        return 0.0

    choice = completion.choices[0]

    # 3) 解析 reward model 的完整输出文本（带 <answer>...）
    msg_content = choice.message.content
    if isinstance(msg_content, list):
        # multi-part content (text + image 等)；RM 实际上只会返回 text，这里保险起见过滤
        texts = [p["text"] for p in msg_content if p.get("type") == "text"]
        rm_response = "".join(texts)
    else:
        rm_response = msg_content

    # 4) 判断 reward model 是否答对（利用你已有的 accuracy_reward 规则）
    rm_acc = accuracy_reward(rm_response, ground_truth)
    if rm_acc <= 0.0:
        # RM 自己都答错了，说明这组 evidence / 问题对它来说不可靠，直接不给视觉 reward
        return 0.0

    # 5) 从 logprobs 中计算生成 token 的平均 logprob
    logprobs_obj = getattr(choice, "logprobs", None)
    avg_logprob: float | None = None

    if logprobs_obj is not None:
        # vLLM OpenAI server 的格式：choice.logprobs.content 是 token 列表
        token_logprobs = []
        content_list = getattr(logprobs_obj, "content", None)

        if content_list is None and isinstance(logprobs_obj, dict):
            # 兼容某些老版本，logprobs 可能是 dict
            content_list = logprobs_obj.get("content")

        if content_list:
            for token_info in content_list:
                # 新版：token_info.logprob; 旧版/字典：token_info["logprob"]
                lp = getattr(token_info, "logprob", None)
                if lp is None and isinstance(token_info, dict):
                    lp = token_info.get("logprob")
                if lp is not None:
                    token_logprobs.append(float(lp))

        if token_logprobs:
            avg_logprob = sum(token_logprobs) / len(token_logprobs)

    # 6) 将平均 logprob 转为 [0,1] 区间的“置信度”
    if avg_logprob is None:
        # logprobs 没返回或解析失败，退化为“只看答对/答错”
        # 此时 rm_acc 要么是 0.0，要么是 1.0（这里已经排除了 0.0 的情况），直接返回 1.0
        return 0.0

    # 几何平均概率：p_geom = exp(avg_logprob) ∈ (0,1]，然后裁剪到 [0,1]
    p_geom = math.exp(avg_logprob)
    if not math.isfinite(p_geom):
        p_geom = 0.0
    p_geom = max(0.0, min(1.0, p_geom))

    # 7) 最终 reward：答错为 0，答对为 [0,1]，越大表示 RM 越有信息
    return float(rm_acc * p_geom)


def compute_score(reward_inputs: list[dict[str, Any]], accuracy_weight: float = 1, format_weight: float = 1, visual_weight: float = 1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []  
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        gt = reward_input["ground_truth"]
        qs = reward_input["question"]
        ### parsing image
        img = _read_raw_image(reward_input["images"])      
        # compute reward
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, gt)
        visual_score = visual_reward(img, qs, response, gt) 
        
        scores.append(
            {
                "overall":  accuracy_weight*accuracy_score + format_weight*format_score + visual_weight*visual_score, 
                "format": format_score,
                "accuracy": accuracy_score,
                "perception": visual_score,
            }
        )

    return scores
