"""Minimal runtime checks for the upgraded PARR module.

Run from the repository root in the same Python environment used for training:

    python -m models.SepReformer_PARR_Libri2Mix_16K.smoke_test_parr
"""

import torch

from .modules.module import ProgressiveAdaptiveResidualRefinement


def main():
    torch.manual_seed(0)

    batch_speakers = 4
    channels = 128
    bottleneck_channels = 64
    time_steps = 65

    parr = ProgressiveAdaptiveResidualRefinement(
        num_stages=4,
        in_channels=channels,
        bottleneck_channels=bottleneck_channels,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout_rate=0.0,
        gamma_init=0.0,
    )
    core = parr.shared_temporal_core

    assert core.temporal_controller.in_channels == 2 * bottleneck_channels
    assert core.temporal_controller.out_channels == bottleneck_channels
    assert torch.count_nonzero(core.temporal_controller.weight).item() == 0
    assert torch.count_nonzero(core.temporal_controller.bias).item() == 0

    bottleneck_outputs = []
    controller_logits = []
    bottleneck_hook = core.bottleneck.register_forward_hook(
        lambda _module, _inputs, output: bottleneck_outputs.append(output.detach())
    )
    controller_hook = core.temporal_controller.register_forward_hook(
        lambda _module, _inputs, output: controller_logits.append(output.detach())
    )

    stage_output = torch.randn(
        batch_speakers, channels, time_steps, requires_grad=True
    )
    refined = parr(stage_output, stage_index=0)
    bottleneck_hook.remove()
    controller_hook.remove()

    assert refined.shape == stage_output.shape
    torch.testing.assert_close(refined, stage_output, rtol=0.0, atol=0.0)
    assert len(bottleneck_outputs) == 1
    assert bottleneck_outputs[0].shape == (
        batch_speakers,
        bottleneck_channels,
        time_steps,
    )
    torch.testing.assert_close(
        bottleneck_outputs[0].mean(dim=1),
        torch.zeros(batch_speakers, time_steps),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert len(controller_logits) == 1
    assert controller_logits[0].shape == (
        batch_speakers,
        bottleneck_channels,
        time_steps,
    )
    assert torch.count_nonzero(controller_logits[0]).item() == 0
    assert torch.all(torch.sigmoid(controller_logits[0]) == 0.5)

    refined.square().mean().backward()
    assert parr.stage_scales[0].grad is not None
    assert torch.isfinite(parr.stage_scales[0].grad)

    parr.zero_grad(set_to_none=True)
    stage_output.grad = None
    with torch.no_grad():
        parr.stage_scales[0].fill_(0.1)

    parr(stage_output, stage_index=0).square().mean().backward()
    controller_grad = core.temporal_controller.weight.grad
    assert controller_grad is not None
    assert torch.isfinite(controller_grad).all()
    assert torch.count_nonzero(controller_grad).item() > 0

    print("PARR smoke test passed.")


if __name__ == "__main__":
    main()
