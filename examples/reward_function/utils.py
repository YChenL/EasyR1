import re
import base64
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

  
# 匹配整个 <box>...</box> 块
_ALL_BOX_PATTERN = re.compile(r"<box>(.*?)</box>", re.DOTALL)

# 匹配严格的 [x1,y1,x2,y2] 形式，允许空白，但不允许多余的 [ ] 或其他内容
_BOX_CONTENT_PATTERN = re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")

def _all_boxes_valid(localize_block: str) -> bool:
    """
    在 <localize>...</localize> 中检查所有 <box>...</box>：
    - 必须至少有一个 <box>...</box>
    - 每个 box 的内容必须是严格的 [x1,y1,x2,y2]（整数），不允许额外的 [ 或 ]
    - 坐标满足 0 <= x1 < x2 <= 1000, 0 <= y1 < y2 <= 1000
    """
    matches = list(_ALL_BOX_PATTERN.finditer(localize_block))

    # 没有任何 <box>...</box>，直接判不合法
    if not matches:
        return False

    for m in matches:
        inner = m.group(1)  # <box> 和 </box> 之间的内容

        coord_match = _BOX_CONTENT_PATTERN.match(inner)
        # 只要有一个不匹配严格格式，整体就不合法
        if not coord_match:
            return False

        x1, y1, x2, y2 = map(int, coord_match.groups())

        # 范围检查：根据你的约定选择 [0, 1000] 或 [0, 999]
        if not (0 <= x1 < x2 <= 1000):
            return False
        if not (0 <= y1 < y2 <= 1000):
            return False

    # 走到这里说明所有 box 都合法
    return True

    
def _extract_ans(response: str) -> str:
    """优先取 <answer>...</answer>，否则回退到 \\boxed{...}。"""
    m = re.search(r"(?s)<answer>(.*?)</answer>", response)
    if m:
        return m.group(1).strip()
    return None

    
def _normalize_yes_no(s: str) -> str | None:
    """
    归一化 Yes/No：
    - 大小写不敏感
    - 去掉首尾空白和结尾标点（如 "Yes."）
    返回 "yes" / "no" 或 None（无法识别为判断题答案）
    """
    if s is None:
        return None
    s = s.strip()
    # 去掉结尾的简单标点
    s = re.sub(r"[.?!\s]+$", "", s)
    s_lower = s.lower()
    if s_lower == "yes":
        return "yes"
    if s_lower == "no":
        return "no"
    return None


def _extract_number_strict(s: str) -> int | None:
    """
    仅当整个字符串都是连续数字时才认为是合法数字答案。
    返回 int 或 None。
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if not re.fullmatch(r"\d+", s):
        return None
    return int(s)


def _normalize_choice_token(tok: str) -> str | None:
    """
    将一个 MC 选项 token 归一化为单个大写字母：
    - 去掉首尾空白
    - 去掉首尾括号 () 和简单标点 ,.;: 等
    - 大小写统一到大写
    - 仅接受 A-Z 这种单个字母，其它一律视为非法（返回 None）
    """
    tok = tok.strip()
    if not tok:
        return None

    # 去掉首尾括号，如 "(A)" -> "A"
    tok = re.sub(r"^\(+", "", tok)
    tok = re.sub(r"\)+$", "", tok)

    tok = tok.strip()

    # 去掉首尾简单标点，比如 'A,' 'A.' '"A"' '[A]'
    tok = re.sub(r'^[\[\]"\'.,;:]+', "", tok)
    tok = re.sub(r'[\[\]"\'.,;:]+$', "", tok)

    tok = tok.strip().upper()

    # 只接受单个字母选项
    if re.fullmatch(r"[A-Z]", tok):
        return tok
    return None


def _parse_mc_options(s: str) -> set[str]:
    """
    解析选择题答案字符串为选项集合（无序）：
    - 支持分隔符：逗号和空白（A,B,C / A B C / (A), (B) 等）
    - 每个 token 归一化为单个大写字母
    - 过滤掉无法归一化的 token
    """
    if s is None:
        return set()
    s = s.strip()
    if not s:
        return set()

    # 按逗号或空白分割：A,B,C / A B C / (A), (B) -> ['A', 'B', 'C'] / ['(A)', '(B)']
    raw_tokens = [t for t in re.split(r"[,\s]+", s) if t]

    normalized: set[str] = set()
    for t in raw_tokens:
        nt = _normalize_choice_token(t)
        if nt is not None:
            normalized.add(nt)

    return normalized


def _looks_like_base64(s: str) -> bool:
    """简单判断一个字符串是否像 base64 图像内容。"""
    s = s.strip()
    if not s:
        return False

    # data URI 形式，直接认为是 base64
    if s.startswith("data:image"):
        return True

    # 太短的一般不是图像的 base64
    if len(s) < 60:
        return False

    # 有空白字符的一般不是纯 base64
    if any(c.isspace() for c in s):
        return False

    # base64 允许的字符集合
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", s):
        return False

    return True


def _image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """将 PIL.Image 对象编码为 base64 字符串（不带 data:image/... 前缀）。"""
    buf = BytesIO()
    img.save(buf, format=format)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def _read_raw_image(input) -> str:
    """
    将输入统一转换为 base64 字符串：
    - 如果 input 是 base64 字符串：直接返回
    - 如果 input 是 PIL.Image.Image 对象：编码为 base64 后返回
    - 如果 input 是路径（str 或 Path）：读取图像后编码为 base64 返回
    """
    # 1. PIL.Image 对象
    if isinstance(input, Image.Image):
        return _image_to_base64(input)

    # 2. 路径 / 字符串
    if isinstance(input, (str, Path)):
        # Path 直接当作路径
        if isinstance(input, Path):
            path = input
        else:
            s = input.strip()

            # 2.a 已经是 base64 字符串
            if _looks_like_base64(s):
                return s

            # 2.b 否则认为是路径
            path = Path(s)

        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        img = Image.open(path).convert("RGB")
        return _image_to_base64(img)

    # 3. 其他类型不支持
    raise TypeError(
        f"_read_raw_image only supports base64 str, PIL.Image.Image, or path-like (str/Path), "
        f"but got type: {type(input)}"
    )


def _parse_bbox_from_output(pred: str):
    """
    仅解析严格格式为 <box>[x1,y1,x2,y2]</box> 的 bbox，
    返回 [[x1,y1,x2,y2], ...]（float）。
    """
    num_pattern = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    # 严格匹配 <box>[x1,y1,x2,y2]</box>
    pattern = (
        rf"<box>\s*\[\s*({num_pattern})\s*,\s*({num_pattern})\s*,"
        rf"\s*({num_pattern})\s*,\s*({num_pattern})\s*\]\s*</box>"
    )

    all_boxes = []
    for m in re.finditer(pattern, pred):
        x1, y1, x2, y2 = map(float, m.groups())
        if x1 < x2 and y1 < y2:
            all_boxes.append([x1, y1, x2, y2])
    return all_boxes


def _crop_image_from_base64(b64_str: str, bbox_list: list, return_pil: bool = False):
    """
    根据给定的 bbox，在 base64 编码的原图上裁剪子图。

    约定：
    - bbox_list 中每个 bbox 为 [x1, y1, x2, y2]
    - 坐标是 0~999 的相对坐标（共 1000 个离散值），允许是 float 或 int
    - 通过 (coord / 999.0) * width/height 得到绝对像素坐标

    参数:
    - return_pil=False（默认）: 返回子图的 base64(JPEG) 字符串列表
    - return_pil=True          : 返回子图的 PIL.Image 列表
    """
    evidence_list = []
    if not bbox_list:
        return evidence_list

    # 解码原图
    try:
        img_bytes = base64.b64decode(b64_str)
    except Exception:
        # base64 有问题直接返回空
        return evidence_list

    with Image.open(BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        width, height = img.size

        for b in bbox_list:
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue

            x1_rel, y1_rel, x2_rel, y2_rel = b
            try:
                x1_rel = float(x1_rel)
                y1_rel = float(y1_rel)
                x2_rel = float(x2_rel)
                y2_rel = float(y2_rel)
            except (TypeError, ValueError):
                continue

            # 将 0~999 相对坐标映射到 [0,1]
            x1_norm = max(0.0, min(1.0, x1_rel / 999.0))
            x2_norm = max(0.0, min(1.0, x2_rel / 999.0))
            y1_norm = max(0.0, min(1.0, y1_rel / 999.0))
            y2_norm = max(0.0, min(1.0, y2_rel / 999.0))

            # 转为像素坐标
            left = int(round(x1_norm * width))
            right = int(round(x2_norm * width))
            top = int(round(y1_norm * height))
            bottom = int(round(y2_norm * height))

            left = max(0, min(width - 1, left))
            top = max(0, min(height - 1, top))
            right = max(0, min(width, right))
            bottom = max(0, min(height, bottom))

            if right <= left or bottom <= top:
                continue

            crop = img.crop((left, top, right, bottom))

            if return_pil:
                # 复制一份，避免离开 with 作用域后图像被关闭
                evidence_list.append(crop.copy())
            else:
                buffer = BytesIO()
                crop.save(buffer, format="JPEG")
                crop_bytes = buffer.getvalue()
                crop_b64 = base64.b64encode(crop_bytes).decode("utf-8")
                evidence_list.append(crop_b64)

    return evidence_list


def _extract_analysis_and_localize(text: str):
    """
    从文本中拆出：
    - analysis_text: <localize> 之前的所有内容
    - localize_body: <localize> 与 </localize> 之间的内容
    """
    if not text:
        return "", ""

    loc_start = text.find("<localize>")
    if loc_start == -1:
        # 没找到 localize，就全部当 analysis 用
        return text, ""

    analysis_text = text[:loc_start]

    loc_end = text.find("</localize>", loc_start)
    if loc_end == -1:
        localize_body = text[loc_start + len("<localize>"):]
    else:
        localize_body = text[loc_start + len("<localize>"): loc_end]

    return analysis_text, localize_body


def _extract_explanations_from_localize(localize_body: str):
    """
    给定 <localize>...</localize> 内部的文本，把每个 <box>[x1,y1,x2,y2]</box>
    后面跟着的一段自然语言解释切出来。

    假定结构类似：
        <box>[..]</box> explanation_for_box1
        <box>[..]</box> explanation_for_box2
        ...

    则返回:
        [explanation_for_box1, explanation_for_box2, ...]
    """
    if not localize_body:
        return []

    pattern = r"<box>\s*\[[^\]]+\]\s*</box>"
    matches = list(re.finditer(pattern, localize_body, flags=re.DOTALL))

    explanations = []
    if not matches:
        return explanations

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(localize_body)
        expl = localize_body[start:end].strip()
        explanations.append(expl)

    return explanations


def _build_message(img: str, question: str, response: str):
    """
    基于：
    - 原始图像 base64: img
    - 原问题: question
    - Round1 输出: response（包含 <localize> 和 <box>[..]</box>）

    构造适配 OpenAI / vLLM-OpenAI-Server 的多模态 messages：
    - system: 单段 text
    - user: content 为 [image_url, text, image_url, text, ..., text(question)]
    """

    # 1) 解析 bbox，并从原图裁剪 evidence 子图（base64 格式）
    bbox_list = _parse_bbox_from_output(response)
    evidence_set = _crop_image_from_base64(img, bbox_list)  # List[str]，每个是 crop 的 base64

    # 2) 提取 <localize> 内文本，按每个 <box> 后面的自然语句作为解释
    _, localize_body = _extract_analysis_and_localize(response)
    explanations = _extract_explanations_from_localize(localize_body)

    # 3) 对齐 crop 图与解释
    paired_evidence = []
    n = min(len(evidence_set), len(explanations))
    for i in range(n):
        paired_evidence.append((evidence_set[i], explanations[i]))
    # 多出来的 crop 没解释就给空串
    for i in range(n, len(evidence_set)):
        paired_evidence.append((evidence_set[i], ""))

    # 4) 组装 user.content：interleaved image + text
    content = []
    for idx, (b64_crop, expl) in enumerate(paired_evidence):
        # image 部分：用 data URL 传给 vLLM OpenAI Server
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_crop}",
            },
        })
        # 对应的文字解释（如果有）
        if expl.strip():
            content.append({
                "type": "text",
                "text": expl.strip(),
            })

    # 最后把原始问题拼上去
    content.append({
        "type": "text",
        "text": question.strip(),
    })

    system_prompt = (
        "You are a helpful assistant. Answer the user's question based on the "
        "interleaved visual evidence with accompanying descriptions. After the "
        "reasoning process, wrap ONLY the final answer between <answer> and "
        "</answer> tags."
    )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    return messages