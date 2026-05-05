import torch
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from torch.utils.data import Dataset, IterableDataset
from torch import nn
from transformers.utils import (
    logging,
)
import transformers
from transformers.utils import is_peft_available
if is_peft_available():
    from peft import PeftModel
from packaging import version
import importlib.metadata
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import sys
import os
sys.path.append(os.path.dirname(__file__))
from Torchmemory import TorchKVMemory
logger = logging.get_logger(__name__)
max_return = 512

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False
class MyTrainer(transformers.Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        torchKVMemory = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if torchKVMemory is None:
            self.torchKVMemory = (TorchKVMemory(128,16),TorchKVMemory(128,16),TorchKVMemory(128,16),TorchKVMemory(128,16))
        else:
            self.torchKVMemory = torchKVMemory
    def compute_loss(self, model, inputs, mem_key_states=None, mem_value_states=None, long_query_states=None, return_outputs=False, return_kv=False, return_query=False, long_mem=None, add_rag=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, all_mem_key_states=mem_key_states,all_mem_value_states=mem_value_states, all_long_query_states=long_query_states, all_long_mem=long_mem, add_rag=add_rag)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if return_kv:
            new_mem_key_states = outputs['mem_key_states']
            new_mem_value_states = outputs['mem_value_states']
        
        long_query_states = outputs["mem_query_states"]

        result = [loss]

        if return_outputs:
            result.append(outputs)
        if return_kv:
            result.append(new_mem_key_states)
            result.append(new_mem_value_states)
        if return_query:
            result.append(long_query_states)
        
        return tuple(result)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs['input_ids'].shape[1]
        position_ids = torch.arange(self.args.segment_length + max_return, device=inputs['input_ids'].device).unsqueeze(0).expand(inputs['input_ids'].shape[0], -1)
        chunks = []
        for start in range(0, seq_len, self.args.segment_length):
            end = min(start + self.args.segment_length, seq_len)
            chunk = {
                k: (v[:, start:end] if v.dim()==2 else v)
                for k, v in inputs.items()
            }
            chunk["position_ids"] = position_ids
            chunks.append(chunk)

        losses = []
        mem_idx = 0
        if self.args.bptt_depth == 0:
            raise ValueError("bptt_depth 不能为 0，请设置为 >= 1 的值。")
        if len(chunks) <= self.args.bptt_depth:
            raise ValueError("len(chunks)不能小于等于bptt_depth")
        for idx in range(len(chunks)):
            mem_key_states = None
            mem_value_states = None
            long_query_states = None
            if idx - self.args.bptt_depth <=0:
                for i in range(idx):
                    if i < mem_idx:
                        with self.compute_loss_context_manager():
                            loss = self.compute_loss(model,chunks[i], mem_key_states=mem_key_states, mem_value_states=mem_value_states, long_query_states=long_query_states, return_kv=True, return_query=True,long_mem=self.torchKVMemory)
                        mem_key_states = loss[1]
                        mem_value_states = loss[2]
                        long_query_states = loss[3]
                    else:
                        with self.compute_loss_context_manager():
                            loss = self.compute_loss(model, chunks[i], mem_key_states=mem_key_states, mem_value_states=mem_value_states, long_query_states=long_query_states,return_kv=True, return_query=True,long_mem=self.torchKVMemory,add_rag = True)
                        mem_key_states = loss[1]
                        mem_value_states = loss[2]
                        long_query_states = loss[3]
                        mem_idx += 1
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, chunks[idx], mem_key_states=mem_key_states, mem_value_states=mem_value_states, long_query_states=long_query_states,long_mem=self.torchKVMemory,add_rag=True)[0]
                mem_idx += 1
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                self.accelerator.backward(loss)
                losses.append(loss.detach())
            else:
                for i in range(self.args.bptt_depth):
                    if idx-self.args.bptt_depth+i < mem_idx:
                        with self.compute_loss_context_manager():
                            loss = self.compute_loss(model,chunks[idx-self.args.bptt_depth+i], mem_key_states=mem_key_states, mem_value_states=mem_value_states, long_query_states=long_query_states, return_kv=True, return_query=True,long_mem=self.torchKVMemory)
                        mem_key_states = loss[1]
                        mem_value_states = loss[2]
                        long_query_states = loss[3]
                    else:
                        with self.compute_loss_context_manager():
                            loss = self.compute_loss(model, chunks[idx-self.args.bptt_depth+i], mem_key_states=mem_key_states, mem_value_states=mem_value_states, long_query_states=long_query_states,return_kv=True, return_query=True,long_mem=self.torchKVMemory,add_rag = True)
                        mem_key_states = loss[1]
                        mem_value_states = loss[2]
                        long_query_states = loss[3]
                        mem_idx += 1
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, chunks[idx], mem_key_states=mem_key_states, mem_value_states=mem_value_states, long_query_states=long_query_states,long_mem=self.torchKVMemory,add_rag=True)[0]
                mem_idx += 1
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                self.accelerator.backward(loss)
                losses.append(loss.detach())
        self.torchKVMemory[0].reset()
        self.torchKVMemory[1].reset()
        self.torchKVMemory[2].reset()
        self.torchKVMemory[3].reset()                         
        
        losses_tensor = torch.stack(losses)  # shape: (n_chunks,)
        valid_losses = losses_tensor[~torch.isnan(losses_tensor)]  # 只保留非 NaN 值

        if len(valid_losses) > 0:
            mean_loss = valid_losses.mean()
        else:
            mean_loss = torch.tensor(0.0, device=losses_tensor.device)
            logger.warning("All losses are NaN for this step.")

        # mean_loss = torch.stack(losses).mean()
        if self.state.global_step % self.args.logging_steps == 0:
            print(f"self.state.global_step= {self.state.global_step}, self.args.logging_steps = {self.args.logging_steps}")
            for idx,loss_item in enumerate(losses):
                self.log({"loss":f"第 {idx} 块的loss : {loss_item.detach().cpu().item()}"})

        del inputs

        return mean_loss / self.args.gradient_accumulation_steps