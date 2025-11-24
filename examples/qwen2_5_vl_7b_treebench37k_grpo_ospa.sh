#!/usr/bin/env bash
set -euxo pipefail

# 只用 0/1/2 做训练，避免占用 3
export CUDA_VISIBLE_DEVICES=0,1

# 让训练侧能访问 judge 服务
export LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY:-EMPTY}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:18901/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-Qwen2.5-72B-Instruct}"

# 你的 7B 模型本地路径
MODEL_PATH="/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/model/TreeVGR-7B-CI"

python3 -m verl.trainer.main \
  config=examples/config_treebench.yaml \
  data.train_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/data/vstar30k_visdrone6k_x1y1x2y2.parquet \
  data.val_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/test-vstar30k_visdrone6k_x1y1x2y2.parquet \
  data.image_dir=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/ \
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
  trainer.experiment_name=qwen2_5_vl_7b_treebench_grpo_llm_as_judge_corrected_ospa \
  worker.reward.reward_function=./examples/reward_function/ospa.py:compute_score \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=25
    
