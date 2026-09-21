"""Read-only validation diagnostics; no persistent hooks or RNG changes."""

import torch
from utils.runtime_state import capture_runtime_state, restore_runtime_state


def core_gradient_norm(model):
    parr = getattr(getattr(model, 'separator', None), 'parr', None)
    if parr is None:
        return None
    gradients = [p.grad.detach().float().square().sum()
                 for p in parr.shared_temporal_core.parameters() if p.grad is not None]
    return torch.stack(gradients).sum().sqrt().item() if gradients else 0.0


def collect_parr_diagnostics(model, dataset, device, count=4, mvn=False):
    parr = getattr(getattr(model, 'separator', None), 'parr', None)
    if parr is None or count <= 0:
        return {}
    from utils.functions import apply_cmvn
    state = capture_runtime_state()
    was_training = model.training
    totals = [0.0] * 5  # count, sum, square sum, low count, high count
    ratios = []

    def gate_hook(module, inputs, logits):
        gate = logits.detach().double().sigmoid()
        values = [gate.numel(), gate.sum().item(), gate.square().sum().item(),
                  (gate < .05).sum().item(), (gate > .95).sum().item()]
        for i, value in enumerate(values):
            totals[i] += value

    def correction_hook(module, inputs, output):
        feature = inputs[0].detach().float()
        ratios.append(((output.detach().float() - feature).norm() /
                       feature.norm().clamp_min(1e-12)).item())

    hooks = []
    try:
        hooks.append(parr.shared_temporal_core.temporal_controller.register_forward_hook(gate_hook))
        hooks.append(parr.register_forward_hook(correction_hook))
        model.eval()
        with torch.inference_mode():
            for i in range(min(count, len(dataset))):
                example = dataset[i]  # deterministic validation center crop; no loader iterator
                waveform = torch.as_tensor(example['mix'], dtype=torch.float32, device=device)[None]
                if mvn:
                    waveform = apply_cmvn(waveform)
                model(waveform, return_aux=False)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
        restore_runtime_state(state)
    n, total, squares, low, high = totals
    if not n:
        raise RuntimeError('PARR diagnostics did not observe any gates.')
    result = {f'parr/gamma_stage_{i}': value.detach().item()
              for i, value in enumerate(parr.stage_scales)}
    result.update({'parr/gate_mean': total / n,
                   'parr/gate_std': max(0., squares / n - (total / n) ** 2) ** .5,
                   'parr/gate_below_005': low / n, 'parr/gate_above_095': high / n,
                   'parr/correction_feature_ratio': sum(ratios) / len(ratios)})
    return result
