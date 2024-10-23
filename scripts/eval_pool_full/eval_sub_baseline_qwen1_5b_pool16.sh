GPUS=$1

# 设置PYTHONPATH环境变量
export PYTHONPATH="./:$PYTHONPATH"

# 运行Python脚本
CUDA_VISIBLE_DEVICES=${GPUS} python eval/mvbench/inference/infer.py \
    --model-path results/pool_full/finetune_qwen1b_poolbranch_pool16 \
    --output-dir json_result/finetune_qwen1b_poolbranch_pool16 \
    --model-base .cache/qwen2_5/1.5b_instruction \
    --conv-mode qwen2_cap \
    --pool_level 16