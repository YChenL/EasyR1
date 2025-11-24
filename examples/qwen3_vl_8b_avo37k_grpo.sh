#!/usr/bin/env bash
set -euxo pipefail

# 只用 0/1/2 做训练，避免占用 3
export CUDA_VISIBLE_DEVICES=0,1

# 让训练侧能访问 judge 服务
export LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY:-EMPTY}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:8000/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-Qwen3-VL-8B-Instruct}"

# 你的8B模型本地路径
MODEL_PATH="/mnt/shared-storage-user/solution/gongyuning/models/Qwen3-VL-8B-Instruct"

python3 -m verl.trainer.main \
  config=examples/config_avo.yaml \
  data.train_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/data/avo_rl_43k.parquet@train[:90%] \
  data.val_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/data/avo_rl_43k.parquet@train[90%:] \
  data.image_dir=/mnt/shared-storage-user/solution/gongyuning/LLaVA-NeXT-Data/images \
  data.target_key=null \
  data.rollout_batch_size=512 \
  data.val_batch_size=512 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=true \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.global_batch_size=256 \
  worker.actor.micro_batch_size_per_device_for_update=4 \
  worker.actor.micro_batch_size_per_device_for_experience=8 \
  worker.rollout.n=8 \
  worker.rollout.gpu_memory_utilization=0.75 \
  worker.rollout.tensor_parallel_size=1 \
  trainer.experiment_name=qwen3_vl_8b_avo_grpo \
  worker.reward.reward_function=./examples/reward_function/avo.py:compute_score \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=10
    
