"""Reproduce selected PARR audit findings without third-party dependencies.

Run from any directory: python docs/parr_review_2026_09_10_probe.py
This is an equation/source audit, NOT a PyTorch forward/backward test.
"""

import ast
import cmath
import copy
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fft(values):
    if len(values) == 1:
        return values
    even, odd = fft(values[::2]), fft(values[1::2])
    twiddled = [cmath.exp(-2j * math.pi * k / len(values)) * value
                for k, value in enumerate(odd)]
    return ([a + b for a, b in zip(even, twiddled)]
            + [a - b for a, b in zip(even, twiddled)])


def magnitude(signal, batch_length, frame_length=1024, hop=256):
    """Same frame coverage, periodic Hann, scale and epsilon as STFT.forward."""
    padded_length = math.ceil(batch_length / hop) * hop
    signal = signal + [0.0] * (padded_length - len(signal))
    scale = 0.5 * math.sqrt(frame_length * frame_length / hop)
    window = [math.sqrt(2 / 3) * (0.5 - 0.5 * math.cos(2 * math.pi * k / frame_length))
              / scale for k in range(frame_length)]
    result = []
    for start in range(0, padded_length - frame_length + 1, hop):
        spectrum = fft([signal[start + k] * window[k] for k in range(frame_length)])
        result.extend(math.sqrt(abs(value) ** 2 + 1e-10)
                      for value in spectrum[:frame_length // 2 + 1])
    return result


def auxiliary_pair_loss(estimate, reference, batch_length):
    """One speaker/pair in PIT_SISNR_mag, before permutation reduction."""
    def centered(x):
        mean = sum(x) / len(x)
        return [v - mean for v in x]
    estimate, reference = centered(estimate), centered(reference)
    scale = max(sum(a * b for a, b in zip(estimate, reference))
                / (sum(v * v for v in reference) + 1e-12), 1e-2)
    a = magnitude(estimate, batch_length)
    b = magnitude([scale * v for v in reference], batch_length)
    return -20 * math.log10(1e-12 + math.sqrt(sum(v * v for v in b))
                           / (math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) + 1e-12))


def main():
    files = [path for directory in ('models', 'utils', 'scripts')
             for path in (ROOT / directory).rglob('*.py')]
    for path in files:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

    # Execute only the actual pure-Python hashing helpers from the repository.
    source = ast.parse((ROOT / 'utils/paired_initialization.py').read_text(encoding='utf-8'))
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_shared_config', '_config_sha256')]
    namespace = dict(copy=copy, hashlib=hashlib, json=json)
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<hash helpers>', 'exec'), namespace)
    config = {'model': {'module_separator': {'parr': {
        'dilations': [1, 2, 4, 8], 'dropout_rate': 0.05}}}}
    changed = copy.deepcopy(config)
    changed['model']['module_separator']['parr'].update(
        dilations=[1, 2, 4, 16], dropout_rate=0.2)
    hash_ignores_parr = namespace['_config_sha256'](config) == namespace['_config_sha256'](changed)
    assert hash_ignores_parr

    length = 1536
    reference = [math.sin(2 * math.pi * 19 * k / 1024)
                 + 0.3 * math.sin(2 * math.pi * 51 * k / 1024) for k in range(length)]
    estimate = [value + (0.8 * math.sin(2 * math.pi * 83 * k / 1024) if k >= 1200 else 0)
                for k, value in enumerate(reference)]
    alone = auxiliary_pair_loss(estimate, reference, length)
    padded = auxiliary_pair_loss(estimate, reference, 2560)
    assert abs(alone - padded) > 0.1

    parameter_components = {
        'bottleneck': 8385, 'two_conv_u_branches': 9088,
        'dense_memory': 62981, 'correction_norm': 128,
        'controller': 8256, 'output_projection': 8320,
        'four_stage_norms_and_scales': 1028,
    }
    output = {
        'scope': 'stdlib equation/source audit; no PyTorch runtime',
        'python_syntax_files_passed': len(files),
        'shared_hash_ignores_dilation_and_dropout_change': hash_ignores_parr,
        'stft_pair_loss_same_valid_samples_db': {
            'valid_length': length, 'batch_length_1536': alone,
            'batch_length_2560': padded, 'difference': padded - alone,
            'frame_counts': [3, 7],
        },
        'stft_512_sample_input': {'padded_length': 512, 'kernel_length': 1024,
                                'valid_convolution_possible': False},
        'silence_estimate_si_snr_from_evaluator_formula_db': 10 * math.log10(1e-8 / 1e-8),
        'parr_parameter_components_analytical': parameter_components,
        'parr_parameters_analytical': sum(parameter_components.values()),
        'parr_convolution_macs_4s_batch1_two_speakers_analytical': 95360 * 15000 * 2,
        'parr_longest_temporal_path_frames': 1 + 2 + 2 * sum([1, 2, 4, 8]),
    }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
