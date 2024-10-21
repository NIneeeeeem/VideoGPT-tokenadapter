# 激活conda环境

GPUS=$1
conda activate videogpt

# 切换工作目录
cd /hhd2/wxc/VideoGPT-plus

# 设置PYTHONPATH环境变量
export PYTHONPATH="/hhd2/wxc/VideoGPT-plus:$PYTHONPATH"

# 运行Python脚本
CUDA_VISIBLE_DEVICES=${GPUS} python /hhd2/wxc/VideoGPT-plus/eval/mvbench/inference/infer.py \
    --model-path results/finetune_qwen1b_baseline_sub_pool8 \
    --output-dir json_result/finetune_sub_qwen15_baseline_pool8 \
    --model-base .cache/qwen2_5/1.5b_instruction \
    --conv-mode qwen2_cap