"""Reference-based evaluation for single-channel speech separation.

The evaluator selects one speaker permutation with SI-SNR and reuses that
assignment for every metric. This avoids reporting an independently optimized
permutation for each metric.
"""

import csv
import json
import os
from importlib.metadata import PackageNotFoundError, version
from itertools import permutations

import numpy as np
from mir_eval.separation import bss_eval_sources

try:
    from pesq import pesq
except ImportError:  # pragma: no cover - exercised in the training environment
    pesq = None

try:
    from pystoi import stoi
except ImportError:  # pragma: no cover - exercised in the training environment
    stoi = None


HIGHER_IS_BETTER_METRICS = (
    "si_snr",
    "si_snri",
    "snr",
    "snri",
    "bss_sdr",
    "bss_sdri",
    "bss_sir",
    "bss_siri",
    "bss_sar",
    "bss_sari",
    "pesq_wb",
    "pesq_wb_i",
    "stoi",
    "stoi_i",
    "estoi",
    "estoi_i",
)

LOWER_IS_BETTER_METRICS = ("mixture_consistency_error_db",)
EFFICIENCY_METRICS = ("latency_seconds", "rtf", "peak_vram_mb")
METRIC_PROTOCOL = "separation-v2-unit-si-snr"
SILENT_ESTIMATE_FLOOR_DB = -80.0


def _package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def json_safe(value):
    """Replace non-finite floats so emitted JSON remains standards-compliant."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _as_mono_float64(signal, length=None):
    array = np.asarray(signal, dtype=np.float64).squeeze()
    if array.ndim != 1:
        raise ValueError(f"Expected mono 1-D audio, got shape {array.shape}.")
    if length is not None:
        if int(length) <= 0 or array.size < int(length):
            raise ValueError("Requested audio length must be positive and available in every signal.")
        array = array[: int(length)]
    if array.size < 2:
        raise ValueError("Metrics require at least two audio samples.")
    if not np.isfinite(array).all():
        raise ValueError("Audio contains NaN or infinity.")
    return array


def _zero_mean(signal):
    return signal - np.mean(signal)


def _unit_centered(signal):
    """Normalize before squaring, preserving very small nonzero signal gains."""
    signal = _as_mono_float64(signal)
    peak = np.max(np.abs(signal))
    if peak == 0:
        return None
    signal = _zero_mean(signal / peak)
    norm = np.linalg.norm(signal)
    return None if norm == 0 else signal / norm


def si_snr(estimate, reference, eps=1.0e-8):
    """Unit-normalized zero-mean SI-SNR; constant estimate gets a -80 dB floor.

    A constant reference is undefined and rejected, not silently excluded.
    Epsilon acts on unit energy so near-silent nonconstant audio keeps its gain
    invariance. This is metric protocol v2; do not mix v1 and v2 CSV scores.
    """
    estimate = _as_mono_float64(estimate)
    reference = _as_mono_float64(reference)
    if estimate.shape != reference.shape:
        raise ValueError("Estimate/reference lengths differ.")
    estimate, reference = _unit_centered(estimate), _unit_centered(reference)
    if reference is None:
        raise ValueError("SI-SNR is undefined for a silent/constant reference.")
    if estimate is None:
        return SILENT_ESTIMATE_FLOOR_DB
    target = np.dot(estimate, reference) * reference
    noise = estimate - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def snr(estimate, reference, eps=1.0e-8):
    """Scale-dependent, zero-mean SNR in dB."""
    estimate = _zero_mean(_as_mono_float64(estimate))
    reference = _zero_mean(_as_mono_float64(reference))
    error = estimate - reference
    return float(10.0 * np.log10((np.dot(reference, reference) + eps) / (np.dot(error, error) + eps)))


def best_si_snr_permutation(estimates, references):
    """Return estimate indices ordered by reference and their SI-SNR values."""
    num_sources = len(references)
    if len(estimates) != num_sources:
        raise ValueError("Estimate/reference speaker counts differ.")
    best_permutation = None
    best_scores = None
    best_total = -np.inf
    for permutation in permutations(range(num_sources)):
        scores = [
            si_snr(estimates[permutation[target]], references[target])
            for target in range(num_sources)
        ]
        total = float(np.sum(scores))
        if total > best_total:
            best_total = total
            best_permutation = permutation
            best_scores = scores
    return tuple(best_permutation), np.asarray(best_scores, dtype=np.float64)


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _bootstrap_interval(values, samples, confidence, random_generator):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    bootstrap_means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = random_generator.choice(values, size=values.size, replace=True)
        bootstrap_means[index] = np.mean(draw)
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(bootstrap_means, [alpha, 1.0 - alpha]).tolist())


class SeparationMetricsEvaluator:
    """Accumulate utterance-level quality and efficiency measurements."""

    def __init__(
        self,
        sampling_rate,
        num_spks,
        compute_bss_eval=True,
        compute_pesq=True,
        compute_stoi=True,
        compute_estoi=True,
        bootstrap_samples=2000,
        confidence=0.95,
        bootstrap_seed=0,
    ):
        self.sampling_rate = int(sampling_rate)
        self.num_spks = int(num_spks)
        self.compute_bss_eval = bool(compute_bss_eval)
        self.compute_pesq = bool(compute_pesq)
        self.compute_stoi = bool(compute_stoi)
        self.compute_estoi = bool(compute_estoi)
        self.bootstrap_samples = int(bootstrap_samples)
        self.confidence = float(confidence)
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between 0 and 1.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be nonnegative (zero disables CI).")
        self.bootstrap_seed = int(bootstrap_seed)
        self.rows = []
        self.failures = []

        if self.compute_pesq:
            if self.sampling_rate != 16000:
                raise ValueError("WB-PESQ requires 16 kHz audio in this evaluator.")
            if pesq is None:
                raise ImportError("WB-PESQ requested. Install dependency: pip install pesq==0.0.4")
        if (self.compute_stoi or self.compute_estoi) and stoi is None:
            raise ImportError("STOI/ESTOI requested. Install dependency: pip install pystoi==0.4.1")

    def _perceptual_score(self, key, name, reference, estimate, extended=False):
        try:
            if name == "pesq_wb":
                return float(pesq(self.sampling_rate, reference, estimate, mode="wb"))
            return float(stoi(reference, estimate, self.sampling_rate, extended=extended))
        except Exception as error:  # metric libraries reject silence/short utterances
            self.failures.append(
                {"key": str(key), "metric": name, "error": str(error)}
            )
            return float("nan")

    def add_utterance(
        self,
        key,
        mixture,
        references,
        estimates,
        length,
        latency_seconds,
        peak_vram_mb=float("nan"),
    ):
        length = int(length)
        mixture = _as_mono_float64(mixture, length)
        references = [_as_mono_float64(value, length) for value in references]
        estimates = [_as_mono_float64(value, length) for value in estimates]
        if len(references) != self.num_spks or len(estimates) != self.num_spks:
            raise ValueError("Unexpected number of speakers during evaluation.")
        for index, reference in enumerate(references):
            if _unit_centered(reference) is None:
                raise ValueError(f"{key}: source {index} is silent/constant; SI-SNR reference is undefined.")
        for index, estimate in enumerate(estimates):
            if _unit_centered(estimate) is None:
                self.failures.append({"key": str(key), "metric": "si_snr", "estimate_index": index,
                                      "error": "Silent/constant estimate retained with -80 dB floor."})

        permutation, separated_si_snr = best_si_snr_permutation(estimates, references)
        estimates = [estimates[index] for index in permutation]
        mixture_si_snr = np.asarray([si_snr(mixture, ref) for ref in references])
        separated_snr = np.asarray([snr(est, ref) for est, ref in zip(estimates, references)])
        mixture_snr = np.asarray([snr(mixture, ref) for ref in references])

        source_metrics = {
            "si_snr": separated_si_snr,
            "si_snri": separated_si_snr - mixture_si_snr,
            "snr": separated_snr,
            "snri": separated_snr - mixture_snr,
        }

        if self.compute_bss_eval:
            try:
                reference_stack = np.stack(references)
                estimate_stack = np.stack(estimates)
                mixture_stack = np.stack([mixture] * self.num_spks)
                separated_bss = bss_eval_sources(
                    reference_stack, estimate_stack, compute_permutation=False
                )
                mixture_bss = bss_eval_sources(
                    reference_stack, mixture_stack, compute_permutation=False
                )
                for metric_name, result_index in (("bss_sdr", 0), ("bss_sir", 1), ("bss_sar", 2)):
                    separated_value = np.asarray(separated_bss[result_index])
                    mixture_value = np.asarray(mixture_bss[result_index])
                    source_metrics[metric_name] = separated_value
                    source_metrics[f"{metric_name}i"] = separated_value - mixture_value
            except Exception as error:
                self.failures.append({"key": str(key), "metric": "bss_sdr", "error": str(error)})
                for metric_name in (
                    "bss_sdr", "bss_sdri", "bss_sir", "bss_siri", "bss_sar", "bss_sari"
                ):
                    source_metrics[metric_name] = np.full(self.num_spks, np.nan)

        perceptual_specs = []
        if self.compute_pesq:
            perceptual_specs.append(("pesq_wb", False))
        if self.compute_stoi:
            perceptual_specs.append(("stoi", False))
        if self.compute_estoi:
            perceptual_specs.append(("estoi", True))
        for metric_name, extended in perceptual_specs:
            separated = np.asarray(
                [
                    self._perceptual_score(key, metric_name, ref, est, extended)
                    for ref, est in zip(references, estimates)
                ]
            )
            mixture_scores = np.asarray(
                [
                    self._perceptual_score(key, metric_name, ref, mixture, extended)
                    for ref in references
                ]
            )
            source_metrics[metric_name] = separated
            source_metrics[f"{metric_name}_i"] = separated - mixture_scores

        duration_seconds = length / self.sampling_rate
        row = {
            "key": str(key),
            "metric_protocol": METRIC_PROTOCOL,
            "num_samples": length,
            "duration_seconds": duration_seconds,
            "permutation": "-".join(map(str, permutation)),
            "latency_seconds": float(latency_seconds),
            "rtf": float(latency_seconds / max(duration_seconds, 1.0e-12)),
            "peak_vram_mb": float(peak_vram_mb),
            "mixture_consistency_error_db": float(
                10.0
                * np.log10(
                    (
                        np.sum((mixture - np.sum(np.stack(estimates), axis=0)) ** 2)
                        + 1.0e-8
                    )
                    / (np.sum(mixture ** 2) + 1.0e-8)
                )
            ),
        }
        for metric_name, values in source_metrics.items():
            values = np.asarray(values, dtype=np.float64)
            row[metric_name] = _safe_mean(values)
            for speaker_index, value in enumerate(values, start=1):
                row[f"{metric_name}_s{speaker_index}"] = float(value)
        self.rows.append(row)
        return row

    def running_mean(self, metric_name):
        return _safe_mean([row.get(metric_name, np.nan) for row in self.rows])

    def summarize(self, model_parameters=None, metadata=None):
        random_generator = np.random.default_rng(self.bootstrap_seed)
        metric_names = HIGHER_IS_BETTER_METRICS + LOWER_IS_BETTER_METRICS + EFFICIENCY_METRICS
        metrics = {}
        for metric_name in metric_names:
            values = np.asarray([row.get(metric_name, np.nan) for row in self.rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            low, high = _bootstrap_interval(
                finite,
                self.bootstrap_samples,
                self.confidence,
                random_generator,
            )
            metrics[metric_name] = {
                "mean": _safe_mean(finite),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "median": float(np.median(finite)) if finite.size else float("nan"),
                "q25": float(np.quantile(finite, 0.25)) if finite.size else float("nan"),
                "q75": float(np.quantile(finite, 0.75)) if finite.size else float("nan"),
                "ci_low": low,
                "ci_high": high,
                # Compatibility aliases are only meaningful at 95% confidence.
                "ci95_low": low if self.confidence == 0.95 else float("nan"),
                "ci95_high": high if self.confidence == 0.95 else float("nan"),
                "n_valid": int(finite.size),
            }
        return {
            "sampling_rate": self.sampling_rate,
            "num_speakers": self.num_spks,
            "num_utterances": len(self.rows),
            "model_parameters": None if model_parameters is None else int(model_parameters),
            "metadata": metadata or {},
            "assignment_metric": "si_snr",
            "metric_protocol": METRIC_PROTOCOL,
            "silent_estimate_floor_db": SILENT_ESTIMATE_FLOOR_DB,
            "metric_implementations": {
                "si_snr_and_snr": "internal NumPy reference implementation",
                "bss_eval": f"mir-eval {_package_version('mir-eval')}",
                "pesq_wb": f"pesq {_package_version('pesq')}; mode=wb",
                "stoi_and_estoi": f"pystoi {_package_version('pystoi')}",
            },
            "confidence": self.confidence,
            "bootstrap_samples": self.bootstrap_samples,
            "metrics": metrics,
            "metric_failures": self.failures,
        }

    def save(self, output_directory, model_parameters=None, metadata=None):
        os.makedirs(output_directory, exist_ok=True)
        if not self.rows:
            raise RuntimeError("No utterances were evaluated.")
        fieldnames = list(self.rows[0].keys())
        with open(os.path.join(output_directory, "metrics_per_utterance.csv"), "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = self.summarize(
            model_parameters=model_parameters,
            metadata=metadata,
        )
        with open(os.path.join(output_directory, "metrics_summary.json"), "w", encoding="utf-8") as stream:
            json.dump(json_safe(summary), stream, indent=2, ensure_ascii=False, allow_nan=False)
        return summary
