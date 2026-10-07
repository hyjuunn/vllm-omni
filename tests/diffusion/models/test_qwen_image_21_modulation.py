# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import _prepare_block_modulation

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_values", [None, [], [True] * 7, [False, True, False, True, True]])
def test_prepared_modulation_preserves_sample_and_condition_rows(batch, dtype, mask_values):
    generator = torch.Generator().manual_seed(42)
    mask = None if mask_values is None else torch.tensor(mask_values, dtype=torch.bool)
    modulation = torch.randn(batch + (mask is not None), 4 * 128, generator=generator).to(dtype)
    prepared = _prepare_block_modulation(modulation, mask)
    sequence = 3 if mask is None else len(mask_values)
    hidden = torch.randn(batch, sequence, 128, generator=generator).to(dtype)

    # Reconstruct the original block operation for each sample/token. This
    # catches incorrect t=0 selection and moving rounding through scale/gate.
    for sample in range(batch):
        for token in range(sequence):
            row = sample if mask is None or mask[token] else -1
            original = modulation[row].chunk(4)
            for block in range(2):
                scale = prepared[2 * block][sample, 0 if mask is None else token]
                gate = prepared[2 * block + 1][sample, 0 if mask is None else token]
                torch.testing.assert_close(
                    hidden[sample, token] * scale,
                    hidden[sample, token] * (1 + original[2 * block]),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(gate, original[2 * block + 1].tanh(), atol=0, rtol=0)
    assert all(item.shape == (batch, 1 if mask is None else sequence, 128) for item in prepared)
