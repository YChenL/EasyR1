#!/usr/bin/env bash
set -euxo pipefail

# 只用 0/1/2 做训练，避免占用 3
export CUDA_VISIBLE_DEVICES=0,1,2

# 让训练侧能访问 judge 服务
export LLM_AS_A_JUDGE_KEY="${LLM_AS_A_JUDGE_KEY:-EMPTY}"
export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-http://127.0.0.1:18901/v1}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:Qwen2.5-32B-Instruct}"

# 你的 7B 模型本地路径
MODEL_PATH="/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/model/TreeVGR-7B-CI"

python3 -m verl.trainer.main \
    config=examples/config_treebench.yaml \
    data.train_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/vstar30k_visdrone6k_x1y1x2y2_train \
    data.val_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/vstar30k_visdrone6k_x1y1x2y2_test \
    data.image_dir=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/ \
    worker.actor.global_batch_size=192 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.rollout.gpu_memory_utilization=0.8 \
    trainer.experiment_name=qwen2_5_vl_7b_treebench_grpo_llm_as_judge \
    trainer.n_gpus_per_node=3
