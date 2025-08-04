# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import numpy as np
from deepspeed.ops.op_builder import EvoformerAttnBuilder
from deepspeed.accelerator import get_accelerator

kernel_ = None


def _attention(Q, K, V, bias1, bias2):
    assert Q.shape[-3] > 16, "seq_len must be greater than 16"
    O = torch.empty_like(Q, dtype=Q.dtype)
    assert get_accelerator().on_accelerator(Q), "Q must be on cuda"
    assert get_accelerator().on_accelerator(K), "K must be on cuda"
    assert get_accelerator().on_accelerator(V), "V must be on cuda"
    assert get_accelerator().on_accelerator(bias1), "bias1 must be on cuda"
    assert get_accelerator().on_accelerator(bias2), "bias2 must be on cuda"
    global kernel_
    if kernel_ is None:
        kernel_ = EvoformerAttnBuilder().load()
    nheads = Q.shape[-2]
    nq = (Q.shape[-3] + 31) // 32 * 32
    nb = np.prod(Q.shape[:-3])
    lse = torch.empty((nb, nheads, nq), dtype=torch.float32, device=Q.device)
    kernel_.attention(Q, K, V, bias1, bias2, O, lse)
    return O, lse


def attention_bwd(dO, Q, K, V, O, lse, bias1, bias2, bias1_grad, bias2_grad):
    assert max(Q.shape[-1], V.shape[-1]) <= 64, "Hidden size is too large. Need to change kMax to a larger value"
    dQ = torch.empty_like(Q, dtype=Q.dtype)
    dK = torch.empty_like(K, dtype=K.dtype)
    dV = torch.empty_like(V, dtype=V.dtype)
    assert get_accelerator().on_accelerator(dO), "dO must be on cuda"
    assert get_accelerator().on_accelerator(Q), "Q must be on cuda"
    assert get_accelerator().on_accelerator(K), "K must be on cuda"
    assert get_accelerator().on_accelerator(V), "V must be on cuda"
    assert get_accelerator().on_accelerator(O), "O must be on cuda"
    global kernel_
    if kernel_ is None:
        kernel_ = EvoformerAttnBuilder().load()
    delta = torch.empty_like(lse)
    if bias1_grad:
        dB1 = torch.zeros_like(bias1, dtype=torch.float32)
    else:
        dB1 = torch.tensor([], dtype=torch.float32, device=bias1.device)
    if bias2_grad:
        dB2 = torch.zeros_like(bias2, dtype=torch.float32)
    else:
        dB2 = torch.tensor([], dtype=torch.float32, device=bias2.device)
    kernel_.attention_bwd(dO, Q, K, V, O, lse, delta, bias1, bias2, dQ, dK, dV, dB1, dB2)
    return dQ, dK, dV, dB1.to(dO.dtype), dB2.to(dO.dtype)


import torch
from torch import Tensor
from torch.library import Library, impl

# Define a custom library namespace
lib = Library("evoformer_attn", "DEF")

# Define the custom operation signature
lib.define("deepspeed_evoformer_attention(Tensor q, Tensor k, Tensor v, Tensor bias1, Tensor bias2) -> (Tensor, Tensor)")
lib.define("deepspeed_evoformer_attention_bwd(Tensor grad_output, Tensor q, Tensor k, Tensor v, Tensor o, Tensor lse, Tensor bias1, Tensor bias2, bool is_b1_grad, bool is_b2_grad) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")


# CUDA implementation (assuming _attention is your kernel function)
def deepspeed_evoformer_attention_impl(q: Tensor, k: Tensor, v: Tensor, bias1: Tensor, bias2: Tensor) -> tuple[Tensor, Tensor]:
    # Your existing CUDA kernel call
    # Replace with your actual kernel function
    o, lse = _attention(q, k, v, bias1, bias2)  # Adjust based on your kernel's signature
    return o, lse

def deepspeed_evoformer_attention_bwd_impl(grad_output: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, bias1: Tensor, bias2: Tensor, is_b1_grad: bool, is_b2_grad: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # 这里调用你原有的 attention_bwd 函数（CUDA 实现）
    # 假设 attention_bwd 是你的 C/C++ 扩展函数
    return attention_bwd(grad_output, q, k, v, o, lse, bias1, bias2, is_b1_grad, is_b2_grad)

# Register the CUDA implementation
lib.impl("deepspeed_evoformer_attention", deepspeed_evoformer_attention_impl, "CUDA")
lib.impl("deepspeed_evoformer_attention_bwd", deepspeed_evoformer_attention_bwd_impl, "CUDA")

# Meta implementation for tracing
def deepspeed_evoformer_attention_meta(q: Tensor, k: Tensor, v: Tensor, bias1: Tensor, bias2: Tensor) -> tuple[Tensor, Tensor]:
    # Return empty tensors with shapes matching the expected output
    o = torch.empty_like(q)  # [*, L, H, D]
    nheads = q.shape[-2]
    nq = (q.shape[-3] + 31) // 32 * 32
    nb = np.prod(q.shape[:-3])
    lse = torch.empty((nb, nheads, nq), dtype=torch.float32, device=q.device)  # 修正为 [nb, nheads, nq]
    return o, lse

def deepspeed_evoformer_attention_bwd_meta(grad_output: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, bias1: Tensor, bias2: Tensor, is_b1_grad: bool, is_b2_grad: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # meta 实现只需要返回与输入张量形状匹配的空张量
    dQ = torch.empty_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)
    dB1 = torch.empty_like(bias1) if is_b1_grad else None
    dB2 = torch.empty_like(bias2) if is_b2_grad else None
    return dQ, dK, dV, dB1, dB2


# Register the meta implementation
lib.impl("deepspeed_evoformer_attention", deepspeed_evoformer_attention_meta, "Meta")
lib.impl("deepspeed_evoformer_attention_bwd", deepspeed_evoformer_attention_bwd_meta, "Meta")

class EvoformerFusedAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias1=None, bias2=None):
        """
        q, k, v: are in shape [*, L, H, D]
        """
        bias1_ = bias1.contiguous() if bias1 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
        bias2_ = bias2.contiguous() if bias2 is not None else torch.tensor([], dtype=q.dtype, device=q.device)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o, lse = torch.ops.evoformer_attn.deepspeed_evoformer_attention(q, k, v, bias1_, bias2_)
        ctx.save_for_backward(q, k, v, o, lse, bias1_, bias2_)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse, bias1, bias2 = ctx.saved_tensors
        is_b1_grad = bias1.numel() != 0 and ctx.needs_input_grad[3]
        is_b2_grad = bias2.numel() != 0 and ctx.needs_input_grad[4]
        dQ, dK, dV, dB1, dB2 = torch.ops.evoformer_attn.deepspeed_evoformer_attention_bwd(grad_output, q, k, v, o, lse, bias1, bias2, is_b1_grad, is_b2_grad)
        if not is_b1_grad:
            dB1 = None
        if not is_b2_grad:
            dB2 = None
        return dQ, dK, dV, dB1, dB2


def DS4Sci_EvoformerAttention(Q, K, V, biases):
    assert len(biases) <= 2

    if (len(biases) == 0):
        biases.append(None)

    if (len(biases) == 1):
        biases.append(None)

    bias_1_shape = lambda x: (x.shape[0], x.shape[1], 1, 1, x.shape[2])
    bias_2_shape = lambda x: (x.shape[0], 1, x.shape[3], x.shape[2], x.shape[2])

    if biases[0] is not None:
        assert biases[0].shape == bias_1_shape(Q), "bias1 shape is incorrect"

    if biases[1] is not None:
        assert biases[1].shape == bias_2_shape(Q), "bias2 shape is incorrect"

    return EvoformerFusedAttention.apply(Q, K, V, biases[0], biases[1])
