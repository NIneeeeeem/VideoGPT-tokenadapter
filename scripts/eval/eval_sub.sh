# 激活conda环境

GPUS=$1
source activate videogpt

# 切换工作目录
cd /hdd2/wxc/VideoGPT-plus

# 设置PYTHONPATH环境变量
export PYTHONPATH="/hdd2/wxc/VideoGPT-plus:$PYTHONPATH"

# 运行Python脚本
CUDA_VISIBLE_DEVICES=${GPUS} python /hdd2/wxc/VideoGPT-plus/eval/mvbench/inference/infer.py \
    --model-path /hdd2/wxc/VideoGPT-plus/results/videogpt_plus_finetune_sub \
    --output-dir json_result/finetune_sub