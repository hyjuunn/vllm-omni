# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from diffusers.models.normalization import RMSNorm

from vllm_omni.diffusion.layers.rms_norm_rope_interleaved import can_fuse_rms_norm_rope, fused_rms_norm_rope
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_keeps_native_dispatch(dtype):
    x = torch.zeros(1, 17, 3, 128, dtype=dtype)
    weight = torch.ones(128, dtype=dtype)
    freqs = torch.ones(17, 64, dtype=torch.complex64)
    assert not can_fuse_rms_norm_rope(x, weight, freqs)
    with pytest.raises(ValueError, match="Expected CUDA BF16"):
        fused_rms_norm_rope(x, weight, freqs, 1e-6)


@pytest.mark.cpu
def test_fake_output_is_contiguous_for_packed_qkv():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        storage = torch.empty(2, 17, 3, 5, 128, device="cuda:0", dtype=torch.bfloat16)
        x = storage[:, :, 1]
        weight = torch.ones(128, device="cuda:0", dtype=torch.bfloat16)
        freqs = torch.ones(17, 64, device="cuda:0", dtype=torch.complex64)
        result = fused_rms_norm_rope(x, weight, freqs, 1e-6)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
        assert result.is_contiguous()


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.skipif(not current_omni_platform.is_rocm(), reason="ROCm fast path")
@pytest.mark.parametrize("shape", [(1, 0, 3), (1, 17, 3), (2, 513, 16), (1, 4096, 32)])
@pytest.mark.parametrize("qkv_part", [0, 1])
@pytest.mark.parametrize("scale", [0.01, 1.0, 10.0])
def test_matches_diffusers_rounding_and_complex_rope(shape, qkv_part, scale):
    batch, sequence, heads = shape
    generator = torch.Generator().manual_seed(12345)
    storage = torch.randn(batch, sequence, 3, heads, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    x = (storage * scale)[:, :, qkv_part]
    norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(128, generator=generator).to(device="cuda", dtype=torch.bfloat16))
    angles = torch.randn(sequence, 64, generator=generator) * 30
    freqs = torch.polar(torch.ones_like(angles), angles).to("cuda")

    normalized = norm(x)
    pairs = torch.view_as_complex(normalized.float().reshape(batch, sequence, heads, 64, 2))
    expected = torch.view_as_real(pairs * freqs.unsqueeze(1)).flatten(3).to(x.dtype)
    assert can_fuse_rms_norm_rope(x, norm.weight, freqs)
    actual = fused_rms_norm_rope(x, norm.weight, freqs, norm.eps)
    assert actual.is_contiguous()
    assert actual.shape == expected.shape
    if expected.numel():
        delta = actual.float() - expected.float()
        relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float())
        assert relative_l2 < 1e-3
        torch.testing.assert_close(actual, expected, atol=0.0625, rtol=0.02)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.skipif(not current_omni_platform.is_rocm(), reason="ROCm fast path")
def test_unsupported_head_and_weight_dtype_keep_native_dispatch():
    x = torch.empty(1, 17, 3, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    freqs = torch.ones(17, 32, device="cuda", dtype=torch.complex64)
    assert not can_fuse_rms_norm_rope(x, weight, freqs)
    x = torch.empty(1, 17, 3, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.float32)
    freqs = torch.ones(17, 64, device="cuda", dtype=torch.complex64)
    assert not can_fuse_rms_norm_rope(x, weight, freqs)


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.skipif(not current_omni_platform.is_rocm(), reason="ROCm fast path")
@pytest.mark.parametrize("compiled", [False, True])
def test_strided_batch_sequence_and_frequency_table(compiled):
    generator = torch.Generator().manual_seed(7)
    storage = torch.randn(34, 2, 3, 5, 128, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    x = storage[::2, :, 1].transpose(0, 1)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(34, 64, generator=generator)
    freqs = torch.polar(torch.ones_like(angles), angles).to("cuda")[::2]
    norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    normalized = norm(x)
    pairs = torch.view_as_complex(normalized.float().reshape(2, 17, 5, 64, 2))
    expected = torch.view_as_real(pairs * freqs.unsqueeze(1)).flatten(3).to(x.dtype)
    call = fused_rms_norm_rope
    if compiled:
        call = torch.compile(call, backend="aot_eager", fullgraph=True)
    actual = call(x, weight, freqs, 1e-6)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
