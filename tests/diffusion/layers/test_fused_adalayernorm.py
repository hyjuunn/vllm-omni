# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dispatch regressions and exact BF16 stage tests for the gfx1100 Ada tail."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers import fused_adalayernorm as fused
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def supported_metadata(monkeypatch):
    """Exercise the guard on CPU without initializing a GPU context."""
    monkeypatch.setattr(fused, "HAS_TRITON", True)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(fused, "_is_gfx1100", lambda index: index == 0)

    def metadata(shape, strides):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            stride=lambda dim: strides[dim],
            dtype=torch.bfloat16,
            device=torch.device("cuda:0"),
            is_cuda=True,
            requires_grad=False,
        )

    return (
        metadata((1, 7920, 3072), (3072, 3072, 1)),
        metadata((1, 1, 3072), (18432, 3072, 1)),
        metadata((1, 1, 3072), (18432, 3072, 1)),
    )


@pytest.mark.cpu
@pytest.mark.parametrize("batch_stride", [3072, 24330240])
def test_supported_singleton_batch_strides(supported_metadata, batch_stride):
    x, scale, shift = supported_metadata
    x.stride = lambda dim: (batch_stride, 3072, 1)[dim]
    assert fused.can_fuse_adalayernorm(x, scale, shift)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "index,field,value",
    [
        (0, "shape", (2, 7920, 3072)),
        (0, "shape", (1, 0, 3072)),
        (0, "shape", (1, 7920, 1536)),
        (0, "ndim", 2),
        (0, "stride", lambda dim: (24330240, 6144, 1)[dim]),
        (0, "stride", lambda dim: (24330240, 3072, 2)[dim]),
        (0, "is_cuda", False),
        (1, "shape", (1, 7920, 3072)),
        (2, "shape", (3072,)),
        (1, "stride", lambda dim: (6144, 6144, 2)[dim]),
        (2, "device", torch.device("cuda:1")),
        *[(index, "dtype", torch.float32) for index in range(3)],
        *[(index, "requires_grad", True) for index in range(3)],
    ],
)
def test_unsupported_metadata_retains_native(supported_metadata, index, field, value):
    setattr(supported_metadata[index], field, value)
    assert not fused.can_fuse_adalayernorm(*supported_metadata)


@pytest.mark.cpu
@pytest.mark.parametrize("reason", ["compile", "no_triton", "no_hip", "other_arch"])
def test_unsupported_runtime_retains_native(monkeypatch, supported_metadata, reason):
    if reason == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        monkeypatch.setattr(fused, "_is_gfx1100", lambda _: pytest.fail("compile queried device properties"))
    elif reason == "no_triton":
        monkeypatch.setattr(fused, "HAS_TRITON", False)
    elif reason == "no_hip":
        monkeypatch.setattr(torch.version, "hip", None)
    else:
        monkeypatch.setattr(fused, "_is_gfx1100", lambda _: False)
    assert not fused.can_fuse_adalayernorm(*supported_metadata)


@pytest.mark.cpu
def test_architecture_feature_suffix(monkeypatch):
    fused._is_gfx1100.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName="gfx1100:sramecc-:xnack-")
    )
    try:
        assert fused._is_gfx1100(0)
    finally:
        fused._is_gfx1100.cache_clear()


@pytest.mark.cpu
@pytest.mark.parametrize("affine", [False, True])
def test_hip_cpu_fallback_preserves_affine_and_gradients(monkeypatch, affine):
    norm = AdaLayerNorm(3072, elementwise_affine=affine)
    if affine:
        with torch.no_grad():
            norm.layernorm.weight.copy_(torch.linspace(0.5, 1.5, 3072))
            norm.layernorm.bias.copy_(torch.linspace(-0.5, 0.5, 3072))
    monkeypatch.setattr(fused, "_launch_tail", lambda *a, **k: pytest.fail("fallback launched Triton"))
    inputs = [torch.randn(shape, requires_grad=True) for shape in ((1, 2, 3072), (1, 1, 3072), (1, 1, 3072))]
    actual = norm.forward_hip(*inputs)
    expected = norm.forward_native(*inputs)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), inputs)
    assert all(torch.equal(a, e) for a, e in zip(actual_grad, expected_grad))


@pytest.mark.cpu
def test_cpu_bf16_public_helper_fallback():
    x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 64, dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    expected = F.layer_norm(x.float(), (64,), None, None, 1e-5).to(x.dtype) * (1 + scale) + shift
    assert torch.equal(fused.fused_adalayernorm(x, scale, shift, eps=1e-5), expected)


@pytest.mark.cpu
def test_affine_hip_fallback_without_gradients(monkeypatch):
    norm = AdaLayerNorm(3072, elementwise_affine=True)
    monkeypatch.setattr(fused, "can_fuse_adalayernorm", lambda *a: pytest.fail("affine reached fusion guard"))
    x = torch.randn(1, 2, 3072)
    scale = torch.randn(1, 1, 3072)
    shift = torch.randn_like(scale)
    with torch.no_grad():
        norm.layernorm.weight.fill_(2)
        norm.layernorm.bias.fill_(0.25)
        assert torch.equal(norm.forward_hip(x, scale, shift), norm.forward_native(x, scale, shift))


@pytest.mark.cpu
def test_hip_compile_preserves_native_graph(monkeypatch):
    norm = AdaLayerNorm(3072)
    monkeypatch.setattr(fused, "_launch_tail", lambda *a, **k: pytest.fail("compile launched Triton"))
    compiled = torch.compile(norm.forward_hip, backend="eager", fullgraph=True)
    x = torch.randn(1, 2, 3072)
    scale = torch.randn(1, 1, 3072)
    shift = torch.randn_like(scale)
    assert torch.equal(compiled(x, scale, shift), norm.forward_native(x, scale, shift))


@pytest.mark.cpu
def test_supported_hip_dispatch_uses_fused_tail(monkeypatch):
    norm = AdaLayerNorm(3072)
    inputs = [torch.randn(shape) for shape in ((1, 2, 3072), (1, 1, 3072), (1, 1, 3072))]
    sentinel = torch.empty_like(inputs[0])
    calls = []
    monkeypatch.setattr(fused, "can_fuse_adalayernorm", lambda *args: True)

    def candidate(*args, eps):
        calls.append((args, eps))
        return sentinel

    monkeypatch.setattr(fused, "fused_adalayernorm", candidate)
    assert norm.forward_hip(*inputs) is sentinel
    assert len(calls) == 1
    assert all(a is b for a, b in zip(calls[0][0], inputs))
    assert calls[0][1] == norm.eps


@pytest.fixture
def gfx1100_device():
    if not torch.version.hip or not torch.cuda.is_available() or not fused.HAS_TRITON:
        pytest.skip("requires ROCm, Triton and gfx1100")
    device = torch.device("cuda", torch.cuda.current_device())
    if not fused._is_gfx1100(device.index):
        pytest.skip("requires gfx1100")
    return device


@pytest.mark.rocm
@pytest.mark.parametrize("tokens", [1, 8, 7920])
@pytest.mark.parametrize("batch_stride", [3072, 24330240])
def test_gpu_native_normalization_and_tail_bitwise(gfx1100_device, tokens, batch_stride):
    generator = torch.Generator(device=gfx1100_device).manual_seed(42)
    x = torch.randn((1, tokens, 3072), device=gfx1100_device, dtype=torch.bfloat16, generator=generator)
    x = x.as_strided(x.shape, (batch_stride, 3072, 1))
    modulation = torch.randn((1, 6, 3072), device=gfx1100_device, dtype=torch.bfloat16, generator=generator)
    scale, shift = modulation[:, 1:2], modulation[:, 0:1]
    assert scale.storage_offset() == 3072
    assert fused.can_fuse_adalayernorm(x, scale, shift)
    nfp32 = F.layer_norm(x.float(), (3072,), None, None, 1e-6)
    n = nfp32.to(torch.bfloat16)
    a = 1 + scale
    p = n * a
    y = p + shift
    stages = fused._launch_tail(nfp32, scale, shift, debug=True)
    assert all(torch.equal(stages[key], expected) for key, expected in (("a", a), ("p", p), ("y", y)))
    assert torch.equal(fused.fused_adalayernorm(x, scale, shift), y)
    assert torch.equal(AdaLayerNorm(3072).forward_hip(x, scale, shift), y)


@pytest.mark.rocm
def test_gpu_tail_preserves_each_bf16_barrier(gfx1100_device):
    generator = torch.Generator(device=gfx1100_device).manual_seed(7)
    nfp32 = torch.randn((1, 8, 3072), device=gfx1100_device, generator=generator)
    # Exact ties make normalization/scale rounding observable, with random
    # elements covering multiply rounding and cancellation in the final add.
    nfp32[..., :1024] = 1.00390625
    scale = torch.randn((1, 1, 3072), device=gfx1100_device, generator=generator).to(torch.bfloat16)
    shift = torch.randn_like(scale)
    scale[..., :1024] = 0.00390625
    shift[..., :1024] = -0.5
    n = nfp32.to(torch.bfloat16)
    a = 1 + scale
    p = n * a
    y = p + shift
    stages = fused._launch_tail(nfp32, scale, shift, debug=True)
    assert all(torch.equal(stages[key], expected) for key, expected in (("a", a), ("p", p), ("y", y)))
    collapsed = (nfp32 * (1 + scale.float()) + shift.float()).to(torch.bfloat16)
    assert not torch.equal(y, collapsed)
    assert not torch.equal(a, 1 + scale.float())
    assert not torch.equal(p.float(), n.float() * a.float())
    assert not torch.equal(y, (n.float() * a.float() + shift.float()).to(torch.bfloat16))
