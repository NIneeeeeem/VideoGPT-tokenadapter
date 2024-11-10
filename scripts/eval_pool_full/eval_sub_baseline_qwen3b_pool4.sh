GPUS=$1

# 设置PYTHONPATH环境变量
export PYTHONPATH="./:$PYTHONPATH"

# 运行Python脚本
CUDA_VISIBLE_DEVICES=${GPUS} python eval/mvbench/inference/infer.py \
    --model-path results/pool_full/finetune_qwen3b_poolbranch_pool4 \
    --output-dir json_result/finetune_qwen3b_poolbranch_pool4 \
    --model-base .cache/qwen2_5/3b_instruction \
    --conv-mode qwen2_cap \
    --pool_level 4