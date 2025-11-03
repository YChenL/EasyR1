#!/usr/bin/env bash
set -euo pipefail

# ================= 可调参数（与你的 Python 代码保持一致） =================
PORT="${1:-18901}"                               # /v1 所在端口；你的 Python 用 http://<host>:PORT/v1
HOST="0.0.0.0"                                   # 对外可访问
MODEL_PATH="/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/model/Qwen2.5-32B-Instruct"  # 本地模型路径
SERVED_NAME="Qwen2.5-32B-Instruct"               # /v1/models 返回的 id；你的代码里会 fallback 到这个名字
API_KEYS="${API_KEYS:-EMPTY}"                     # 和 Python 里的 LLM_AS_A_JUDGE_KEY 对齐
DTYPE="${DTYPE:-auto}"                            # auto/float16/bfloat16
MAX_LEN="${MAX_LEN:-8192}"                        # 最大上下文；想省显存可改小如 4096
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"                # vLLM 并发序列数，提升吞吐的关键
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"              # vLLM 预留显存比例；单实例建议 0.85~0.92
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/data/hf_models}"   # 缓存目录，可留空

# ================ 固定到单卡 H200: GPU 3 运行 =================
export CUDA_VISIBLE_DEVICES=3

# ================ 可选：HF 镜像/鉴权（若需要） =================
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN=xxxxxxxxxxxxxxxx

# ================ 依赖（首次需要） =================
# pip install -U "vllm>=0.5.0"

# ================ 启动 vLLM OpenAI 兼容服务 =================
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --download-dir "${DOWNLOAD_DIR}" \
  --trust-remote-code \
  --api-keys "${API_KEYS}"
  # 可选稳定性/占用优化项（按需追加）：
  # --enforce-eager
  # --kv-cache-dtype fp8        # Hopper 支持；更省显存（如遇不兼容就去掉）
  # --max-num-batched-tokens 4096
