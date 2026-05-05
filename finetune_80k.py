# Portions of this file are adapted from LongLoRA:
# https://github.com/JIA-Lab-research/LongLoRA
# The original LongLoRA code is licensed under the Apache License 2.0.
# Modifications were made for Training-Inference Consistent Segmented Execution.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugpy attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print(f"debugpy exception: {e}")
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional
import torch
import transformers


from model import LlamaForCausalLMWithMemory
from trainer import MyTrainer
from transformers import set_seed
import datasets
import transformers.models.llama.modeling_llama as modeling_llama


def _dynamic_frequency_update_patched(self, position_ids, device):
    # position_ids: [bs, seq_len]
    seq_len = int((torch.max(position_ids) + 1).item())

    rope_scaling = getattr(self.config, "rope_scaling", None)
    min_scale_len = None
    if rope_scaling is not None:
        min_scale_len = rope_scaling.get("min_scale_len", None)

    # clamp short seq to long-scale frequency space
    if min_scale_len is not None:
        seq_len_eff = max(seq_len, int(min_scale_len))
    else:
        seq_len_eff = seq_len

    # grow: recompute inv_freq using effective seq_len
    if seq_len_eff > self.max_seq_len_cached:
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config,
            device,
            seq_len=seq_len_eff,
            **self.rope_kwargs,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len_eff

    # reset: also based on effective seq_len (important!)
    if seq_len_eff < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len


# install patch (must be BEFORE from_pretrained)
modeling_llama.LlamaRotaryEmbedding._dynamic_frequency_update = _dynamic_frequency_update_patched
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    data_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    bptt_depth: Optional[int] = field(default=0, metadata={"help": "max number of previous segments in gradient computation."})
    max_length: Optional[int] = field(default=8192,metadata={"help": "输入最大长度"})
    segment_length: Optional[int] = field(default=4096, metadata={"help": "segment length"})

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def collate_truncate(batch, max_token_length):
    input_ids=[]
    labels=[]
    for example in batch:
        example['input_ids'] = example['input_ids'][:max_token_length]
        input_ids.append(example['input_ids'])
        labels.append([example['labels']])
    new_batch = {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(input_ids)
    }
    return new_batch
def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    print(f"随机种子为:  {training_args.seed}")
    set_seed(training_args.seed)
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.rope_scaling = {
        "rope_type": "dynamic",
        "factor": 10.0,
        "min_scale_len": 8192,
    }

    # Load model and tokenizer
    model = LlamaForCausalLMWithMemory.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    print("Tokenizer vocab size:", len(tokenizer))
    dataset = datasets.load_dataset(training_args.data_dir, split="train")
    dataset = dataset.select_columns(['input_ids', 'labels'])
    dataset = dataset.shuffle(seed=training_args.seed)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = MyTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=partial(collate_truncate, max_token_length=training_args.model_max_length))
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
