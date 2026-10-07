# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native FP32 LayerNorm with a BF16 modulation tail for eager gfx1100.

The native reduction is deliberately retained. Each BF16 cast is an observable
rounding boundary in ``BF16(LN_FP32(x)) * BF16(1 + scale) + shift``.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

_WIDTH = 3072
_BLOCK_SIZE = 1024


@lru_cache(maxsize=None)
def _is_gfx1100(device_index: int) -> bool:
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx1100"


def can_fuse_adalayernorm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    """Accept only BF16 batch-one rows with one broadcast modulation vector.

    Singleton strides and storage offsets do not affect pointer indexing. This
    contract also permits Wan norm_out when it has the same layout as norm1/3.
    """
    if torch.compiler.is_compiling() or not HAS_TRITON or not torch.version.hip:
        return False
    if any(t.requires_grad for t in (x, scale, shift)):
        return False
    if not x.is_cuda or any(t.device != x.device or t.dtype != torch.bfloat16 for t in (x, scale, shift)):
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] == 0 or x.shape[2] != _WIDTH:
        return False
    if x.stride(-2) != _WIDTH or x.stride(-1) != 1:
        return False
    if any(t.shape != (1, 1, _WIDTH) or t.stride(-1) != 1 for t in (scale, shift)):
        return False
    return _is_gfx1100(x.device.index)


if HAS_TRITON:

    @triton.jit
    def _adalayernorm_tail_kernel(
        norm_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        a_ptr,
        p_ptr,
        numel,
        debug: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        channels = offsets % 3072
        norm = tl.load(norm_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + channels, mask=mask, other=0).to(tl.float32)
        shift = tl.load(shift_ptr + channels, mask=mask, other=0).to(tl.float32)
        # Explicit round-to-nearest-even at every native BF16 boundary.
        n = norm.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        a = (1.0 + scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        p = (n * a).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        y = (p + shift).to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(out_ptr + offsets, y, mask=mask)
        if debug:
            tl.store(a_ptr + channels, a, mask=mask & (offsets < 3072))
            tl.store(p_ptr + offsets, p, mask=mask)


def _launch_tail(
    norm: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, debug: bool = False
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Launch the production arithmetic; debug only adds intermediate stores."""
    out = torch.empty(norm.shape, dtype=torch.bfloat16, device=norm.device)
    a = torch.empty_like(scale) if debug else out
    p = torch.empty_like(out) if debug else out
    with torch.cuda.device(norm.device):
        _adalayernorm_tail_kernel[(triton.cdiv(norm.numel(), _BLOCK_SIZE),)](
            norm,
            scale,
            shift,
            out,
            a,
            p,
            norm.numel(),
            debug=debug,
            block_size=_BLOCK_SIZE,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return {"a": a, "p": p, "y": out} if debug else out


def fused_adalayernorm(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Non-affine AdaLayerNorm; unsupported inputs retain native PyTorch math."""
    if not can_fuse_adalayernorm(x, scale, shift):
        return F.layer_norm(x.float(), (x.shape[-1],), None, None, eps).to(x.dtype) * (1 + scale) + shift
    norm = F.layer_norm(x.float(), (_WIDTH,), None, None, eps)
    return _launch_tail(norm, scale, shift)
