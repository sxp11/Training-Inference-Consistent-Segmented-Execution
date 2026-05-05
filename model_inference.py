from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


from transformers.generation import GenerationMixin
from transformers.cache_utils import Cache, DynamicCache

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,

)

from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    repeat_kv,
)
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from dataclasses import dataclass
logger = logging.get_logger(__name__)
max_return = 512
inference_len = 64

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_only_K(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

class LocalLlamaFlashAttention2(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "selected_heads",
            torch.tensor(
                [0,1,2,4,9,12,14,15,16,18,19,22,23,26,29,30],
                dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "other_heads",
            torch.tensor(
                [3,5,6,7,8,10,11,13,17,20,21,24,25,27,28,31],
                dtype=torch.long
            ),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        mem_key_states: Optional[torch.Tensor] = None,
        mem_value_states: Optional[torch.Tensor] = None,
        return_kv = False,
        is_generate = False,
        is_first_generate = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], torch.Tensor, torch.Tensor]:

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = key_states.clone()
            value_states = value_states.clone()


        chunk_len = key_states.shape[2]
        if return_kv:
            return_mem_key_states = key_states[:,self.other_heads,-max_return:,].clone()
            return_mem_value_states = value_states[:,self.other_heads,-max_return:,].clone()
        else:
            return_mem_key_states = None
            return_mem_value_states = None
        if mem_key_states is None:
            if is_generate and not is_first_generate:
                query_states, key_states[:,:,-1:] = apply_rotary_pos_emb(query_states, key_states[:,:,-1:], cos[:,chunk_len-1:chunk_len], sin[:,chunk_len-1:chunk_len])
                key_states[:,:,:-1] = apply_rotary_pos_emb_only_K(key_states[:,:,:-1], cos[:,:chunk_len-1], sin[:,:chunk_len-1])
                q, indices, cu_q_lens, max_s = unpad_input(query_states.transpose(1,2), attention_mask[:,-1:])
                kv = torch.stack([key_states, value_states], dim=2)
                kv, _, cu_k_lens, max_k = unpad_input(kv.transpose(1,3),attention_mask)
                output_unpad = flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_q_lens,
                    cu_k_lens,
                    max_s,
                    max_k,
                    0.0,
                    softmax_scale=None,
                    causal=True,
                )
                output = rearrange(
                    pad_input(
                        output_unpad.reshape(-1,self.num_heads * self.head_dim), indices, bsz, q_len
                    ),
                    "b s (h d) -> b s h d",
                    h = self.num_heads,
                )
                output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)
                return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value, return_mem_key_states, return_mem_value_states
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:,:chunk_len], sin[:,:chunk_len])
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                qkv = torch.stack(
                    [query_states, key_states, value_states], dim=2
                )  # [bsz, nh, 3, q_len, hd]

                qkv = qkv.transpose(1, 3)
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_q_lens, max_s = unpad_input(x, attention_mask)
                x_unpad = rearrange(
                    x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
                )
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
                )
                output = rearrange(
                    pad_input(
                        rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
                    ),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
                output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

                return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value, return_mem_key_states, return_mem_value_states

        else:
            if is_generate and not is_first_generate:
                query_attention_mask = attention_mask[:,-1:]
                query_states, key_states[:,:,-1:] = apply_rotary_pos_emb(query_states, key_states[:,:,-1:], cos[:,max_return+chunk_len-1:max_return+chunk_len], sin[:,max_return+chunk_len-1:max_return+chunk_len])
                key_states[:,:,:-1] = apply_rotary_pos_emb_only_K(key_states[:,:,:-1], cos[:,max_return:max_return+chunk_len-1], sin[:,max_return:max_return+chunk_len-1])
            else:
                query_attention_mask = attention_mask
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:,max_return:max_return+chunk_len], sin[:,max_return:max_return+chunk_len])
            mem_key_states = mem_key_states.to(query_states.dtype)
            mem_value_states = mem_value_states.to(query_states.dtype)
            mem_key_states = apply_rotary_pos_emb_only_K(mem_key_states, cos[:,:max_return], sin[:,:max_return])
            extra_attention_mask = F.pad(attention_mask, (max_return, 0), value=1)
            total_len = max_return + chunk_len  # chunk_len==q_len 通常
            second_kv = key_states.new_empty((bsz, self.num_key_value_heads//2, 2, total_len, self.head_dim))
            first_kv = key_states.new_empty((bsz, self.num_key_value_heads//2, 2, chunk_len, self.head_dim))
            # prefix
            second_kv[:, :, 0, :max_return, :] = mem_key_states
            second_kv[:, :, 1, :max_return, :] = mem_value_states
            # chunk
            second_kv[:, :, 0, max_return:, :] = key_states[:,self.other_heads]
            second_kv[:, :, 1, max_return:, :] = value_states[:,self.other_heads]


            second_q, second_indices, second_cu_q_lens_r, second_max_s = unpad_input(query_states[:,self.other_heads].transpose(1,2), query_attention_mask)

            
            second_kv, _, second_cu_kv_lens, second_kv_max_s = unpad_input(second_kv.transpose(1, 3),extra_attention_mask)
            second_output_unpad = flash_attn_varlen_kvpacked_func(
                second_q,
                second_kv,
                second_cu_q_lens_r,
                second_cu_kv_lens,
                second_max_s,
                second_kv_max_s,
                0.0,
                softmax_scale=None,
                causal=True,
            )
            second_output = rearrange(
                pad_input(
                    second_output_unpad.reshape(-1,self.num_heads//2 * self.head_dim),second_indices, bsz, q_len
                ),
                "b s (h d) -> b s h d",
                h = self.num_heads//2,
            )
            first_kv[:, :, 0] = key_states[:,self.selected_heads]
            first_kv[:, :, 1] = value_states[:,self.selected_heads]


            first_q, first_indices, first_cu_q_lens_r, first_max_s = unpad_input(query_states[:,self.selected_heads].transpose(1,2), query_attention_mask)

            
            first_kv, _, first_cu_kv_lens, first_kv_max_s = unpad_input(first_kv.transpose(1, 3),attention_mask)
            first_output_unpad = flash_attn_varlen_kvpacked_func(
                first_q,
                first_kv,
                first_cu_q_lens_r,
                first_cu_kv_lens,
                first_max_s,
                first_kv_max_s,
                0.0,
                softmax_scale=None,
                causal=True,
            )
            first_output = rearrange(
                pad_input(
                    first_output_unpad.reshape(-1,self.num_heads//2 * self.head_dim),first_indices, bsz, q_len
                ),
                "b s (h d) -> b s h d",
                h = self.num_heads//2,
            )
            output = torch.empty(
                (bsz, q_len, self.num_key_value_heads, self.head_dim),
                device=query_states.device,
                dtype=query_states.dtype
                )
            output.index_copy_(2, self.selected_heads, first_output)
            output.index_copy_(2, self.other_heads, second_output)
            return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value, return_mem_key_states, return_mem_value_states
        

class LongLlamaFlashAttention2(LlamaFlashAttention2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "selected_heads",
            torch.tensor(
                [0,1,2,4,9,12,14,15,16,18,19,22,23,26,29,30],
                dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "other_heads",
            torch.tensor(
                [3,5,6,7,8,10,11,13,17,20,21,24,25,27,28,31],
                dtype=torch.long
            ),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        mem_key_states: Optional[torch.Tensor] = None,
        mem_value_states: Optional[torch.Tensor] = None,
        long_mem = None,
        long_query_states: Optional[torch.Tensor] = None,
        add_rag = False,
        return_kv = False,
        is_generate = False,
        is_first_generate = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            assert isinstance(past_key_value, DynamicCache)
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states = key_states.clone()
            value_states = value_states.clone()

        chunk_len = key_states.shape[2]
        if return_kv:
            return_mem_key_states = key_states[:,self.other_heads,-max_return:,].detach().clone()
            return_mem_value_states = value_states[:,self.other_heads,-max_return:,].detach().clone()
        else:
            return_mem_key_states = None
            return_mem_value_states = None
        if is_generate:
            if long_query_states is not None:
                return_long_query_states = torch.cat([long_query_states, query_states[:,self.selected_heads]], dim=2)[:,:,-inference_len:].detach().clone()
            else:
                return_long_query_states = query_states[:,self.selected_heads,-inference_len:].detach().clone()
            long_query_states = return_long_query_states
        else:
            return_long_query_states = query_states[:,self.selected_heads, -inference_len:].detach().clone()
        if mem_key_states is None:
            if add_rag:
                long_mem.add(key_states[:,self.selected_heads].detach(), value_states[:,self.selected_heads].detach())
            if is_generate and not is_first_generate:
                query_states, key_states[:,:,-1:] = apply_rotary_pos_emb(query_states, key_states[:,:,-1:], cos[:,chunk_len-1:chunk_len], sin[:,chunk_len-1:chunk_len])
                key_states[:,:,:-1] = apply_rotary_pos_emb_only_K(key_states[:,:,:-1], cos[:,:chunk_len-1], sin[:,:chunk_len-1])
                q, indices, cu_q_lens, max_s = unpad_input(query_states.transpose(1,2), attention_mask[:,-1:])
                kv = torch.stack([key_states, value_states], dim=2)
                kv, _, cu_k_lens, max_k = unpad_input(kv.transpose(1,3),attention_mask)
                output_unpad = flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_q_lens,
                    cu_k_lens,
                    max_s,
                    max_k,
                    0.0,
                    softmax_scale=None,
                    causal=True,
                )
                output = rearrange(
                    pad_input(
                        output_unpad.reshape(-1,self.num_heads * self.head_dim), indices, bsz, q_len
                    ),
                    "b s (h d) -> b s h d",
                    h = self.num_heads,
                )
                output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)
                return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value, return_mem_key_states, return_mem_value_states, return_long_query_states
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:,:chunk_len], sin[:,:chunk_len])
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                qkv = torch.stack(
                    [query_states, key_states, value_states], dim=2
                )  # [bsz, nh, 3, q_len, hd]

                qkv = qkv.transpose(1, 3)
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_q_lens, max_s = unpad_input(x, attention_mask)
                x_unpad = rearrange(
                    x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
                )
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
                )
                output = rearrange(
                    pad_input(
                        rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
                    ),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
                output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

                return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value, return_mem_key_states, return_mem_value_states, return_long_query_states


        else:
            mem_key_states = mem_key_states.to(query_states.dtype)
            mem_value_states = mem_value_states.to(query_states.dtype)
            total_len = max_return + chunk_len
            final_key_states   = key_states.new_empty((bsz, self.num_key_value_heads, total_len, self.head_dim))
            final_value_states = value_states.new_empty((bsz, self.num_key_value_heads, total_len, self.head_dim))
            k_tensor, v_tensor = long_mem.search(long_query_states.detach(), top_k=16, max_return=max_return)
            k_tensor = k_tensor.to(device=query_states.device, dtype=query_states.dtype)
            v_tensor = v_tensor.to(device=query_states.device, dtype=query_states.dtype)
            if add_rag:
                long_mem.add(key_states[:,self.selected_heads].detach(), value_states[:,self.selected_heads].detach())
            final_key_states[:,:,max_return:] = key_states
            final_value_states[:,:,max_return:] = value_states
            final_key_states[:,self.selected_heads,:max_return] = k_tensor
            final_value_states[:,self.selected_heads,:max_return] = v_tensor
            final_key_states[:,self.other_heads,:max_return] = mem_key_states
            final_value_states[:,self.other_heads,:max_return] = mem_value_states
            if is_generate and not is_first_generate:
                query_attention_mask = attention_mask[:,-1:]
                query_states, final_key_states[:,:,max_return+chunk_len-1:max_return+chunk_len] = apply_rotary_pos_emb(query_states, final_key_states[:,:,max_return+chunk_len-1:max_return+chunk_len], cos[:,max_return+chunk_len-1:max_return+chunk_len], sin[:,max_return+chunk_len-1:max_return+chunk_len])
                final_key_states[:,:,max_return:max_return+chunk_len-1] = apply_rotary_pos_emb_only_K(final_key_states[:,:,max_return:max_return+chunk_len-1], cos[:,max_return:max_return+chunk_len-1], sin[:,max_return:max_return+chunk_len-1])
            else:
                query_attention_mask = attention_mask
                query_states, final_key_states[:,:,max_return:] = apply_rotary_pos_emb(query_states, final_key_states[:,:,max_return:], cos[:,max_return:max_return+chunk_len], sin[:,max_return:max_return+chunk_len])
            
            final_key_states[:,:,:max_return] = apply_rotary_pos_emb_only_K(final_key_states[:,:,:max_return], cos[:,:max_return], sin[:,:max_return])
            
            extra_attention_mask = F.pad(attention_mask, (max_return, 0), value=1)
            # extra_mask = torch.ones((bsz, max_return), dtype=attention_mask.dtype, device=attention_mask.device)
            # extra_attention_mask = torch.cat([extra_mask, attention_mask], dim=-1) 
            kv = torch.stack([final_key_states
                            ,final_value_states], dim=2)
            q, indices, cu_q_lens_r, max_s = unpad_input(query_states.transpose(1,2), query_attention_mask)
            kv, _, cu_kv_lens, kv_max_s = unpad_input(kv.transpose(1, 3),extra_attention_mask)
            output_unpad = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q_lens_r,
                cu_kv_lens,
                max_s,
                kv_max_s,
                0.0,
                softmax_scale=None,
                causal=True,
            )
            output = rearrange(
                pad_input(
                    output_unpad.reshape(-1,self.num_heads * self.head_dim),indices, bsz, q_len
                ),
                "b s (h d) -> b s h d",
                h = self.num_heads,
            )
            
            return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value, return_mem_key_states, return_mem_value_states, return_long_query_states
        
class LocalLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LocalLlamaFlashAttention2(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        mem_key_states = None,
        mem_value_states = None,
        return_kv = False,
        is_generate = False,
        is_first_generate = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, mem_key_states, mem_value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            mem_key_states=mem_key_states,
            mem_value_states=mem_value_states,
            return_kv=return_kv,
            is_generate=is_generate,
            is_first_generate=is_first_generate,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        outputs += (mem_key_states, mem_value_states,)

        return outputs

class LongLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LongLlamaFlashAttention2(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        mem_key_states = None,
        mem_value_states = None,
        long_mem = None,
        long_query_states = None,
        add_rag = False,
        return_kv = False,
        is_generate = False,
        is_first_generate = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, mem_key_states, mem_value_states, mem_query_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            mem_key_states=mem_key_states,
            mem_value_states=mem_value_states,
            long_mem=long_mem,
            long_query_states=long_query_states,
            add_rag=add_rag,
            return_kv=return_kv,
            is_generate=is_generate,
            is_first_generate=is_first_generate,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (mem_key_states, mem_value_states, mem_query_states)

        return outputs

@dataclass
class RAGMemBaseModelOutputWithPast(BaseModelOutputWithPast):
    mem_key_states: Optional[Tuple[torch.Tensor]] = None
    mem_value_states: Optional[Tuple[torch.Tensor]] = None
    mem_query_states: Optional[Tuple[torch.Tensor]] = None
class LlamaModelWithMemory(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if i in (6, 8, 11, 16):
                self.layers.append(LongLlamaDecoderLayer(config, i))
            else:
                self.layers.append(LocalLlamaDecoderLayer(config, i))


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        all_mem_key_states = None,
        all_mem_value_states = None,
        all_long_mem = None,
        all_long_query_states = None,
        add_rag = False,
        return_kv = False,
        is_generate = False,
        is_first_generate = False,
    ) -> Union[Tuple, RAGMemBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = attention_mask
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_mem_key = ()
        all_mem_value = ()
        all_mem_query = ()
        query_idx = 0
        for idx, decoder_layer in enumerate(self.layers):
            if idx in(6, 8, 11, 16):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                
                mem_key_states = all_mem_key_states[idx] if all_mem_key_states is not None else None
                mem_value_states = all_mem_value_states[idx] if all_mem_value_states is not None else None
                long_query_states = all_long_query_states[query_idx] if all_long_query_states is not None else None
                long_mem = all_long_mem[query_idx] if all_long_mem is not None else None
                query_idx += 1
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        mem_key_states,
                        mem_value_states,
                        long_mem,
                        long_query_states,
                        add_rag,
                        return_kv,
                        is_generate,
                        is_first_generate,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        mem_key_states=mem_key_states,
                        mem_value_states=mem_value_states,
                        long_mem=long_mem,
                        long_query_states=long_query_states,
                        add_rag=add_rag,
                        return_kv=return_kv,
                        is_generate=is_generate,
                        is_first_generate=is_first_generate,
                    )

                hidden_states = layer_outputs[0]
                output_idx = 1
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                    output_idx += 1
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    output_idx += 1
                all_mem_key += (layer_outputs[output_idx],)
                all_mem_value += (layer_outputs[output_idx+1],)
                all_mem_query += (layer_outputs[output_idx+2].detach(),)
            else:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                
                mem_key_states = all_mem_key_states[idx] if all_mem_key_states is not None else None
                mem_value_states = all_mem_value_states[idx] if all_mem_value_states is not None else None
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        mem_key_states,
                        mem_value_states,
                        return_kv,
                        is_generate,
                        is_first_generate,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        mem_key_states=mem_key_states,
                        mem_value_states=mem_value_states,
                        return_kv=return_kv,
                        is_generate=is_generate,
                        is_first_generate=is_first_generate,
                    )

                hidden_states = layer_outputs[0]
                output_idx = 1
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                    output_idx += 1
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    output_idx += 1
                all_mem_key += (layer_outputs[output_idx],)
                all_mem_value += (layer_outputs[output_idx+1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return RAGMemBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mem_key_states=all_mem_key,
            mem_value_states=all_mem_value,
            mem_query_states=all_mem_query,
        )

@dataclass
class RAGMemCausalLMOutputWithPast(CausalLMOutputWithPast):
    mem_key_states: Optional[Tuple[torch.Tensor]] = None
    mem_value_states: Optional[Tuple[torch.Tensor]] = None
    mem_query_states: Optional[Tuple[torch.Tensor]] = None
class LlamaForCausalLMWithMemory(LlamaForCausalLM, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWithMemory(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        all_mem_key_states = None,
        all_mem_value_states = None,
        all_long_mem = None,
        all_long_query_states = None,
        add_rag = False,
        return_kv = False,
        is_generate = False,
        is_first_generate = False,
    ) -> Union[Tuple, RAGMemCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            all_mem_key_states=all_mem_key_states,
            all_mem_value_states=all_mem_value_states,
            all_long_mem=all_long_mem,
            all_long_query_states=all_long_query_states,
            add_rag=add_rag,
            return_kv=return_kv,
            is_generate=is_generate,
            is_first_generate=is_first_generate,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return RAGMemCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mem_key_states=outputs.mem_key_states,
            mem_value_states=outputs.mem_value_states,
            mem_query_states=outputs.mem_query_states,
        )
    def prepare_inputs_for_generation(
        self,
        input_ids,
        my_attention_mask,
        is_first_generate,
        is_generate,
        mypast_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        position_ids = torch.arange(max_return + input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        if is_generate and not is_first_generate:
            input_ids = input_ids[:, -1:]
            # position_ids = position_ids[:,-1:]
        
        model_inputs = {"input_ids": input_ids}


        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": mypast_key_values,
                "use_cache": use_cache,
                "attention_mask": my_attention_mask,
            }
        )
        return model_inputs