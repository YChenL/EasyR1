Revisions
1.新增examples/config_ours.yaml, 对应我们方法的configs
2.新增examples/format_prompt/treebench.jinja, 对应我们方法的系统提示词
3*.新增examples/reward_function/treebench.py, 对应原文方法的奖励函数
4*.新增examples/reward_function/agent.py, 支持llm as a judge
5*.新增examples/reward_function/affine.py, 支持一个基于仿射变换构建的visual reward
6.新增examples/qwen2_5_7b_treebench_grpo.sh,  对应我们方法的训练脚本
7.修改verl/trainer/data_loader.py, 以支持treebench数据集的视觉监督内容
8.修改verl/utils/dataset.py,  以支持treebench数据集的视觉监督内容，支持llm as a judge
9.修改verl/trainer/config.py, 以支持treebench数据集的视觉监督内容
10.修改verl/workers/reward/function.py, 以支持treebench数据集的视觉监督内容，支持llm as a judge

Quick Start
Step1: 把数据集https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K下载到/data/TreeVGR-RL-37K (里面应该包含一个jsonl的数据集文件和一个images文件夹用来存图像源文件)

Step2: 把模型权重https://huggingface.co/HaochenWang/TreeVGR-7B-CI下载/model/TreeVGR-7B-CI
       把https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct下载到model/Qwen2.5-32B-Instruct

Step3: 运行server/vllm_server.sh; 拉起vllm服务，维护一个reward model对模型输出进行打分;
       进一个新端口，运行examples/qwen2_5_7b_treebench_grpo.sh

