#!/bin/bash

set -x

MODEL_PATH=/model/TreeVGR-7B-CI # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_ours.yaml \
    data.train_files=/data/TreeVGR-RL-37K/train.jsonl@train[:90%] # replace it with the json file path
    data.val_files=/data/TreeVGR-RL-37K/train.jsonl@train[90%:] # replace it with the json file path
    data.image_dir=/data/TreeVGR-RL-37K # replace it with your local dataset path
    worker.actor.model.model_path=${MODEL_PATH}