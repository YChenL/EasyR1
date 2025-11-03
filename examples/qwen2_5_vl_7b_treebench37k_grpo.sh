#!/bin/bash

set -x

MODEL_PATH=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/model/TreeVGR-7B-CI # replace it with your local file path

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config_ours.yaml \
    data.train_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/data/vstar30k_visdrone6k_x1y1x2y2.parquet@train[:90%] \
    data.val_files=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/data/vstar30k_visdrone6k_x1y1x2y2.parquet@train[90%:] \
    data.image_dir=/mnt/shared-storage-user/solution/gongyuning/rl-center/EasyR1/data/TreeVGR-RL-37K/ \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_treebench_grpo \
    trainer.n_gpus_per_node=4
