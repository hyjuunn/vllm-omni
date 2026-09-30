from contextlib import nullcontext

import pytest
import torch

from vllm_omni.diffusion.distributed.autoencoders import autoencoder_kl_wan as wan_vae_module
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyOmniAutoencoderKLWan(OmniAutoencoderKLWan):
    def __init__(self, *, dtype: torch.dtype):
        torch.nn.Module.__init__(self)
        self.register_parameter("dummy_weight", torch.nn.Parameter(torch.ones(1, dtype=dtype)))


def test_wan_vae_execution_context_handles_fp32():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.float32)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.float32


def test_wan_vae_execution_context_handles_bf16():
    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)
    with model._execution_context():
        output = model.dummy_weight + 1
    assert output.dtype == torch.bfloat16


def test_wan_vae_execution_context_uses_platform_autocast(mocker):
    sentinel = object()
    platform = mocker.Mock()
    platform.create_autocast_context.return_value = sentinel
    mocker.patch.object(wan_vae_module, "current_omni_platform", platform)

    model = _DummyOmniAutoencoderKLWan(dtype=torch.bfloat16)

    assert model._execution_context() is sentinel
    platform.create_autocast_context.assert_called_once_with(
        device_type=model.dummy_weight.device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


class _DummyWanAttentionBlock(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1


def test_enable_wan_vae_math_sdpa_wraps_attention_once(mocker):
    model = torch.nn.Sequential(_DummyWanAttentionBlock(), torch.nn.Identity())
    mocker.patch.object(wan_vae_module, "WanAttentionBlock", _DummyWanAttentionBlock)
    math_context = mocker.patch.object(wan_vae_module, "sdpa_kernel", return_value=nullcontext())

    assert wan_vae_module._enable_wan_vae_math_sdpa(model) == 1
    assert wan_vae_module._enable_wan_vae_math_sdpa(model) == 0
    assert torch.equal(model(torch.tensor([1.0])), torch.tensor([2.0]))
    math_context.assert_called_once_with(torch.nn.attention.SDPBackend.MATH)
