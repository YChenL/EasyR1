import re
import os
import random
import requests
from typing import Tuple, List

from openai import OpenAI


# ========== LLM-as-a-judge 客户端初始化 ==========
OPENAI_API_KEY = os.environ.get("LLM_AS_A_JUDGE_KEY", "EMPTY")
BASE_URL = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.3.123:18901/v1")

_client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
try:
    model_name = requests.get(f"{BASE_URL}/models", timeout=5).json()["data"][0]["id"]
except Exception:
    # 回退到显式指定或常用名（需与后端部署一致）
    model_name = os.environ.get("LLM_AS_A_JUDGE_MODEL", "Qwen2.5-72B-Instruct")


# ========== 工具函数 ==========
def _extract_ans(response: str) -> str:
    """优先取 <answer>...</answer>，否则回退到 \\boxed{...}。"""
    m = re.search(r"(?s)<answer>(.*?)</answer>", response)
    if m:
        return m.group(1).strip()
    return None


def _tokenize_mc(s: str) -> Tuple[str, ...]:
    """
    将选项串标准化为有序元组，保留 A-H（大小写不敏感），
    忽略空白/分隔符 ()[]{} ,;/|-& 等。
    例：'(A), c ; B' -> ('A','B','C')
    """
    letters = re.findall(r"[A-Za-z]", s.upper())
    keep = [ch for ch in letters if ch in list("ABCDEFGH")]
    return tuple(sorted(keep))


def _only_contains_mc_tokens(s: str) -> bool:
    """
    若把所有可忽略字符去掉后仅剩 A-H 的若干字母，则判定为“只有选项字母”。
    避免把 'Answer is A because ...' 误判为选择题。
    """
    cleaned = re.sub(r"[\s,;/\-\|&\(\)\[\]\{\}]", "", s.upper())
    return bool(cleaned) and all(ch in "ABCDEFGH" for ch in cleaned)


def _looks_like_mc(a: str, b: str) -> bool:
    """同时满足两端都“只含选项字母”时，视为（多）选题答案。"""
    return _only_contains_mc_tokens(a) and _only_contains_mc_tokens(b)


def get_chat_template():
    chat_template = """
Below are two answers to a question.  [Question] is the question, [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, *Judgement Value* is equal to 1; if they are different, *Judgement Value* is equal to 0. Just output the *Judgement Value* and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
*Judgement Value* is 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : The barrier lies on the right of the picture.
*Judgement Value* is 0
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement Value is 1
""" # noqa

    example_4 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
*Judgement Value* is 1
""" # noqa

    example_5 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
*Judgement Value* is 0
""" # noqa

    example_6 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
*Judgement Value* is 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def _judge_with_llm(ans: str, ground_truth: str, question: str = "") -> float:
    """
    用 Qwen2.5-72B-Instruct 做语义一致性判别：
    等价 -> 'Judgement: 1'；不等价 -> 'Judgement: 0'
    """
    full_prompt = get_prompt(ans, ground_truth, question)

    resp = _client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful, deterministic judge. Output ONLY the *Judgement Value* of 1 or 0.."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.0,              # 固定判分
        seed=0,                       # 可复现
        top_p=1.0,
        max_tokens=25,                 # 限制续写长度，避免多话
    )
    response = resp.choices[0].message.content.strip()

    # print(response)
    acc_reward = 0.0
    verdict = None

    # 解析 "Judgement Value: 1/0"（大小写不敏感；Judgment/Judgement 都接受）
    m = re.search(r'(?i)\bjudg[e]?ment\s*value\b\s*[:=]?\s*([01])\b', response)
    if not m:
        # 兜底：模型只输出 "1" 或 "0"
        m = re.search(r'^\s*([01])\s*$', response)

    if m:
        acc_reward = 1.0 if m.group(1) == '1' else 0.0
    else:
        print(f' [WARNING] resp format error response={response!r}')
        acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(ans) >= 1000:
        acc_reward = -1.0
        # is_format_error = True

    return acc_reward
