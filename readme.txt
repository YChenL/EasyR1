Revisions
1.新增examples/config_ours.yaml, 对应我们方法的configs
2.新增examples/format_prompt/treebench.jinja, 对应我们方法的系统提示词
3*.新增examples/reward_function/treebench.py, 对应原文方法的奖励函数
4.新增examples/qwen2_5_7b_treebench_grpo.sh,  对应我们方法的训练脚本
5.修改verl/trainer/data_loader.py, 以支持treebench数据集的视觉监督内容
6.修改verl/utils/dataset.py,  以支持treebench数据集的视觉监督内容
7.修改verl/trainer/config.py, 以支持treebench数据集的视觉监督内容
8.修改verl/workers/reward/function.py, 以支持treebench数据集的视觉监督内容

Quick Start
Step1: 把数据集https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K下载到/data/TreeVGR-RL-37K (里面应该包含一个jsonl的数据集文件和一个images文件夹用来存图像源文件)
Step2: 把模型权重https://huggingface.co/HaochenWang/TreeVGR-7B-CI下载/model/TreeVGR-7B-CI
Step3: 运行examples/qwen2_5_7b_treebench_grpo.sh