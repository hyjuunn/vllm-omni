# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm RMSNorm plus interleaved complex RoPE with explicit BF16 rounding.

Normalize in FP32, round to BF16, apply the BF16 learned scale and round again,
then rotate pairs using FP32 frequencies. This is the Diffusers RMSNorm ordering;
rounding only after the learned scale implements a different function.
"""

import torch
from torch.library import Library
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.platforms import current_omni_platform

if HAS_TRITON:

    @triton.jit
    def _norm_rope(
        x_ptr,
        weight_ptr,
        freqs_ptr,
        out_ptr,
        stride_batch: tl.constexpr,
        stride_sequence: tl.constexpr,
        stride_head: tl.constexpr,
        freqs_stride: tl.constexpr,
        sequence_length: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        heads = tl.program_id(1) * heads_per_program + tl.arange(0, heads_per_program)
        d = tl.arange(0, head_dim)
        base = (
            (token // sequence_length) * stride_batch
            + (token % sequence_length) * stride_sequence
            + heads[:, None] * stride_head
        )
        x = tl.load(x_ptr + base + d[None, :], heads[:, None] < num_heads, 0).to(tl.float32)
        weight = tl.load(weight_ptr + d).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / head_dim + eps)
        # Diffusers rounds BEFORE the learned scale, then rounds the product.
        norm = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        norm = (norm * weight[None, :]).to(tl.bfloat16).to(tl.float32)
        other_d = d ^ 1
        other_x = tl.load(x_ptr + base + other_d[None, :], heads[:, None] < num_heads, 0).to(tl.float32)
        other_w = tl.load(weight_ptr + other_d).to(tl.float32)
        other_norm = (other_x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
        other_norm = (other_norm * other_w[None, :]).to(tl.bfloat16).to(tl.float32)
        cos = tl.load(freqs_ptr + (token % sequence_length) * freqs_stride + (d // 2) * 2)
        sin = tl.load(freqs_ptr + (token % sequence_length) * freqs_stride + (d // 2) * 2 + 1)
        rotated = tl.where(
            (d % 2) == 0,
            norm * cos[None, :] - other_norm * sin[None, :],
            norm * cos[None, :] + other_norm * sin[None, :],
        )
        tl.store(
            out_ptr + token * num_heads * head_dim + heads[:, None] * head_dim + d[None, :],
            rotated,
            heads[:, None] < num_heads,
        )


def _supported_layout(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.ndim == 4
        and x.shape[-1] == 128
        and x.stride(-1) == 1
        and x.dtype == weight.dtype == torch.bfloat16
        and weight.shape == (128,)
        and weight.is_contiguous()
        and freqs.dtype == torch.complex64
        and freqs.shape == (x.shape[1], 64)
        and freqs.stride(-1) == 1
        and x.device == weight.device == freqs.device
    )


def can_fuse_rms_norm_rope(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor) -> bool:
    """Constrain the fast path; tensor subclasses keep their original dispatch."""
    return (
        HAS_TRITON
        and current_omni_platform.is_rocm()
        and type(x) is torch.Tensor
        and type(weight) in (torch.Tensor, torch.nn.Parameter)
        and type(freqs) is torch.Tensor
        and _supported_layout(x, weight, freqs)
    )


def _rms_norm_rope_impl(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    b, s, h, d = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if b * s * h == 0:
        return out
    table = torch.view_as_real(freqs)
    _norm_rope[(b * s, triton.cdiv(h, 8))](
        x,
        weight,
        table,
        out,
        *x.stride()[:3],
        table.stride(0),
        s,
        h,
        d,
        eps,
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def _rms_norm_rope_fake(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
direct_register_custom_op(
    op_name="rms_norm_rope_interleaved_bf16",
    op_func=_rms_norm_rope_impl,
    fake_impl=_rms_norm_rope_fake,
    mutates_args=[],
    target_lib=_OMNI_OP_LIB,
)


def fused_rms_norm_rope(x: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor, eps: float) -> torch.Tensor:
    """Fuse supported [batch, sequence, heads, 128] BF16 inputs on ROCm.

    Call ``can_fuse_rms_norm_rope`` before choosing this path. Packed QKV views
    and non-contiguous batch/sequence strides are supported; output is contiguous.
    Frequencies are complex64 [sequence, 64], shared across batch and heads.
    """
    if not _supported_layout(x, weight, freqs):
        raise ValueError("Expected CUDA BF16 [B, S, H, 128], BF16 weight [128], and complex64 frequencies [S, 64]")
    return torch.ops.vllm_omni.rms_norm_rope_interleaved_bf16(x, weight, freqs, eps)
