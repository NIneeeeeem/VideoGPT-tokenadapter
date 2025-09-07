# Motivation on Dynamic Compression


## Qwen2.5 - 3B Experiments

- Download [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) weights to `.cache/qwen2_5`
- Pretrained projector weights are available at [Wangxc1000/qwen2.5_3B_projectors](https://huggingface.co/Wangxc1000/qwen2.5_3B_projectors)
- Training script: `scripts/pool_full/finetune_poolbranch_3b_full_pool4.sh`
- Inference script: `scripts/eval_pool_full/eval_sub_baseline_qwen3b_pool4.sh`
- MVBench (token-aware): `$MVBench_{token ~ aware}$`
- To generate a merged JSON file:  
  ```bash
  bash mvbench_hard.sh --pool2 <path_to_pool2_results_folder> --output <output_json_path>
  ```
  The `pool2` folder should contain multiple JSON files; the script merges them into one.


## Pretraining Data & Scripts

- Data: Download [instruction_tuning](https://huggingface.co/datasets/MBZUAI/VideoGPT-plus_Training_Dataset) which includes pretraining images. Unzip in the `pretrain` folder.
- Pretraining scripts:
  ```bash
  bash scripts/pretrain_projector_image_encoder.sh
  bash scripts/pretrain_projector_video_encoder.sh
  ```
- Directory structure:
  ```
  VideoGPT-plus
  ├─ .cache
  │  ├─ instruction_data
  │     ├─ pretraining
  │        ├─ COCO
  │        ├─ cc3M
  ```


## Experimental Results

- VideoGPT+ performance under different pooling settings. LLM used: Qwen2.5 1.5B / 3B

## Code & Resources

- Code: [https://github.com/NIneeeeeem/VideoGPT-tokenadapter.git](https://github.com/NIneeeeeem/VideoGPT-tokenadapter.git)

### Data Download (HuggingFace)

- [MBZUAI/VideoGPT-plus_Training_Dataset](https://huggingface.co/datasets/MBZUAI/VideoGPT-plus_Training_Dataset): Contains videos and original annotations
- [OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench): Contains MVBench annotations and videos

### Model Weights (HuggingFace)

- [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [OpenGVLab/InternVideo2-Stage2_1B-224p-f4](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4)
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Wangxc1000/qwen2.5_1.5B_projectors](https://huggingface.co/Wangxc1000/qwen2.5_1.5B_projectors) (latest projector weights)


## Environment Setup

- PyTorch Lab 2.4.0+cu118
- Flash-attn must be installed separately:  
  `pip install flash-attn --no-build-isolation`
- Other dependencies:  
  `pip install -r requirements.txt`

## Training & Testing Scripts

- Pretraining scripts:
  - `scripts/pretrain_projector_image_encoder.sh`
  - `scripts/pretrain_projector_video_encoder.sh`
- Batch size for VideoGPT+ is set to 256
- Training scripts are located in `scripts/pool_full`
- Set Python path before running:
  ```bash
  export PYTHONPATH="/hhd2/wxc/VideoGPT-plus:$PYTHONPATH"
  ```

## Pooling Logic

```python
def divide_and_round(a, b):
    result = a / b
    if result < 1:
        return 1
    else:
        return round(result)

video_feature: shape=(divide_and_round(16, pool_level), divide_and_round(16, pool_level))
image_feature: shape=(divide_and_round(24, pool_level), divide_and_round(24, pool_level))
```
- `pool1` (i.e., `b=1`) means no pooling is applied.

## Testing

- Testing scripts are located in ```scripts/eval_pool_full```