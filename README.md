# Training-Inference Consistent Segmented Execution for Long-Context LLMs

Official PyTorch implementation of **Training-Inference Consistent Segmented Execution for Long-Context LLMs**.

This repository provides a segmented long-context training and inference framework for LLaMA-style causal language models. The key idea is to make training follow the same segment-level execution semantics used at inference time, reducing the mismatch between full-context training and bounded or segmented long-context inference.

## Overview

Transformer-based LLMs face severe computation and memory costs when generating over long contexts with full attention. Many practical long-context inference methods improve efficiency by using bounded-context or segment-level execution only during inference, while the model is still trained with full-context attention. This creates a mismatch between training and inference.

Training-Inference Consistent Segmented Execution addresses this mismatch by:

- Splitting long sequences into fixed-length segments.
- Carrying KV states across segments.
- Restricting gradient propagation through previous segments according to the segmented inference process.
- Allowing selected attention heads to access historical KV states during the forward pass.
- Using a GPU KV memory buffer to retrieve useful past states efficiently.

## Features

- Segment-level long-context training and inference.
- Customized LLaMA model with memory-aware attention.
- FlashAttention2-based efficient attention computation.
- GPU KV memory buffer for historical key/value states.
- Segment-wise BPTT training through a custom Hugging Face `Trainer`.
- Support for 32K and 80K fine-tuning settings.
- Modified generation utilities for segmented inference.

## Repository Structure

```text
.
|-- finetune_32k.py        # Fine-tuning script for the 32K context setting
|-- finetune_80k.py        # Fine-tuning script for the 80K context setting with RoPE scaling
|-- model.py               # Training-time LLaMA model with segmented memory
|-- model_inference.py     # Inference-time LLaMA model with segmented memory
|-- trainer.py             # Custom Trainer for segmented BPTT training
|-- Torchmemory.py         # GPU KV memory buffer and retrieval module
|-- generation.py          # Modified generation utilities for segmented inference
|-- train.sh               # Example training command
|-- requirements.txt       # Python dependencies
`-- LICENSE
```

## Environment

The experiments were run with the following main environment:

```text
Python 3.10.19
CUDA 11.8
PyTorch 2.2.0+cu118
Transformers 4.45.0
Datasets 2.19.1
Accelerate 0.34.2
FlashAttention 2.5.8
Einops 0.7.0
```

Install PyTorch according to your CUDA version. For the CUDA 11.8 environment used in our experiments:

```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

FlashAttention may require a compatible CUDA toolkit and compiler. If the prebuilt wheel is not available for your platform, install it with:

```bash
pip install flash-attn==2.5.8 --no-build-isolation
```

## Data Preparation

The training data should be a pre-tokenized Hugging Face `datasets`-compatible dataset and should be loadable from `--data_dir` by `datasets.load_dataset`.

Each example is expected to contain tokenized input sequences. The training scripts truncate sequences to `model_max_length` and construct language modeling labels from the same token sequence.

## Training

### 32K Fine-Tuning

```bash
python finetune_32k.py \
  --model_name_or_path /path/to/base_model \
  --data_dir /path/to/tokenized_dataset \
  --output_dir /path/to/output \
  --cache_dir cache \
  --bf16 True \
  --model_max_length 32768 \
  --segment_length 4096 \
  --bptt_depth 1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 5 \
  --tf32 True \
  --report_to tensorboard \
  --ddp_find_unused_parameters False
```

### 80K Fine-Tuning

```bash
python finetune_80k.py \
  --model_name_or_path /path/to/base_model \
  --data_dir /path/to/tokenized_dataset \
  --output_dir /path/to/output \
  --cache_dir cache \
  --bf16 True \
  --model_max_length 81920 \
  --segment_length 4096 \
  --bptt_depth 1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 5 \
  --tf32 True \
  --report_to tensorboard \
  --ddp_find_unused_parameters False
```

## Important Arguments

| Argument | Description |
| --- | --- |
| `model_name_or_path` | Path or Hugging Face name of the base LLaMA-style model. |
| `data_dir` | Dataset path or dataset loading script used by `datasets.load_dataset`. |
| `output_dir` | Directory for checkpoints and the final model. |
| `cache_dir` | Cache directory for model and tokenizer files. |
| `model_max_length` | Maximum sequence length used during training. |
| `segment_length` | Length of each segment in segmented execution. |
| `bptt_depth` | Number of previous segments involved in gradient propagation. |
| `bf16` | Whether to use bfloat16 training. |
| `gradient_accumulation_steps` | Number of gradient accumulation steps. |

## Main Components

### `model.py`

Defines the training-time memory-augmented LLaMA architecture. The main classes include:

- `LocalLlamaFlashAttention2`: customized FlashAttention2 attention module with segment memory support.
- `LlamaModelWithMemory`: LLaMA backbone that passes memory states across layers.
- `LlamaForCausalLMWithMemory`: causal language model wrapper used for fine-tuning.

### `model_inference.py`

Defines the inference-time model variant for segmented generation.

### `trainer.py`

Implements `MyTrainer`, a custom Hugging Face `Trainer` that splits long sequences into segments and performs segment-wise training with controlled gradient propagation.

### `Torchmemory.py`

Implements GPU-side KV memory:

- `MultiHeadExponentialBuffer`: contiguous dynamically growing multi-head buffer.
- `TorchKVMemory`: stores historical KV states and retrieves relevant states for later segments.

### `generation.py`

Contains modified generation utilities for segmented inference, including segment-aware model input preparation and memory passing.

## Notes

- The code assumes a LLaMA-style model architecture.
- FlashAttention2 is required by the customized attention implementation.
- `finetune_80k.py` applies dynamic RoPE scaling for longer context lengths.
- The default segment length used in the provided scripts is `4096`.
- Some scripts set `CUDA_VISIBLE_DEVICES` internally. Please modify it according to your machine before running.

## Citation

If you find this repository useful, please cite our paper. The official PMLR BibTeX will be updated once it becomes available:

```bibtex
@inproceedings{shang2026training,
  title     = {Training-Inference Consistent Segmented Execution for Long-Context {LLMs}},
  author    = {Shang, Xianpeng and Li, Jiang and Duo, Zehua and Cai, Qianyi and Su, Xiangdong},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
  note      = {Accepted at ICML 2026}
}
```

## Acknowledgements

Parts of the fine-tuning code are adapted from [LongLoRA](https://github.com/JIA-Lab-research/LongLoRA). The original LongLoRA code is licensed under the Apache License 2.0.

Parts of the implementation are also inspired by or adapted from [CCA-Attention](https://github.com/chenyaofo/CCA-Attention), the official repository for **Core Context Aware Transformers for Long Context Language Modeling**.

This project also builds on PyTorch, Hugging Face Transformers, Hugging Face Datasets, and FlashAttention.

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.
