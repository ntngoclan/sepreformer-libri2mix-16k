"""Paired statistical comparison of two separation evaluation CSV files."""

import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from .evaluation_metrics import (
    EFFICIENCY_METRICS,
    HIGHER_IS_BETTER_METRICS,
    LOWER_IS_BETTER_METRICS,
    json_safe,
    _bootstrap_interval,
)


def _load_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {row["key"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate utterance key in {path}.")
    return keyed


def _paired_bootstrap(deltas, samples, confidence, random_generator):
    return _bootstrap_interval(deltas, samples, confidence, random_generator)


def _benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full_like(p_values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not finite_indices.size:
        return adjusted
    order = finite_indices[np.argsort(p_values[finite_indices])]
    previous = 1.0
    total = len(order)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = total - reverse_rank + 1
        value = min(previous, p_values[index] * total / rank)
        adjusted[index] = min(value, 1.0)
        previous = value
    return adjusted


def compare(baseline_path, candidate_path, bootstrap_samples, confidence, seed,
            allow_subset=False, expected_count=None):
    if not 0 < confidence < 1 or bootstrap_samples < 0:
        raise ValueError("Require 0 < confidence < 1 and bootstrap_samples >= 0.")
    baseline = _load_rows(baseline_path)
    candidate = _load_rows(candidate_path)
    if not allow_subset and set(baseline) != set(candidate):
        raise ValueError("CSV key sets differ; use --allow-subset only for exploratory comparisons.")
    if expected_count is not None and not allow_subset:
        if len(baseline) != expected_count or len(candidate) != expected_count:
            raise ValueError(f"Expected {expected_count} utterances in both CSV files.")
    common_keys = sorted(set(baseline) & set(candidate))
    if not common_keys:
        raise ValueError("The two CSV files have no common utterance keys.")
    for key in common_keys:
        if baseline[key].get("metric_protocol") != candidate[key].get("metric_protocol"):
            raise ValueError(f"Metric protocol differs for key {key}; re-evaluate both checkpoints.")
        if baseline[key].get("num_samples") != candidate[key].get("num_samples"):
            raise ValueError(f"Audio duration differs for key {key}.")

    random_generator = np.random.default_rng(seed)
    metric_names = HIGHER_IS_BETTER_METRICS + LOWER_IS_BETTER_METRICS + EFFICIENCY_METRICS
    results = []
    raw_p_values = []
    for metric_name in metric_names:
        paired = []
        for key in common_keys:
            if metric_name not in baseline[key] or metric_name not in candidate[key]:
                continue
            base_value = float(baseline[key][metric_name])
            candidate_value = float(candidate[key][metric_name])
            if np.isfinite(base_value) and np.isfinite(candidate_value):
                paired.append((base_value, candidate_value))
        if not paired:
            continue
        values = np.asarray(paired, dtype=np.float64)
        deltas = values[:, 1] - values[:, 0]
        low, high = _paired_bootstrap(
            deltas, bootstrap_samples, confidence, random_generator
        )
        lower_is_better = metric_name in LOWER_IS_BETTER_METRICS + EFFICIENCY_METRICS
        signed_improvement = -deltas if lower_is_better else deltas
        try:
            p_value = float(wilcoxon(deltas, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        raw_p_values.append(p_value)
        results.append(
            {
                "metric": metric_name,
                "direction": "lower" if lower_is_better else "higher",
                "n_pairs": int(values.shape[0]),
                "n_excluded": len(common_keys) - int(values.shape[0]),
                "baseline_mean": float(np.mean(values[:, 0])),
                "candidate_mean": float(np.mean(values[:, 1])),
                "delta_candidate_minus_baseline": float(np.mean(deltas)),
                "delta_ci_low": low,
                "delta_ci_high": high,
                "delta_ci95_low": low if confidence == 0.95 else float("nan"),
                "delta_ci95_high": high if confidence == 0.95 else float("nan"),
                "improved_fraction": float(np.mean(signed_improvement > 0.0)),
                "wilcoxon_p": p_value,
            }
        )

    adjusted = _benjamini_hochberg(raw_p_values)
    for result, adjusted_p in zip(results, adjusted):
        result["wilcoxon_fdr_bh_p"] = float(adjusted_p)
    return {
        "baseline": os.path.abspath(baseline_path),
        "candidate": os.path.abspath(candidate_path),
        "num_common_utterances": len(common_keys),
        "baseline_only_keys": sorted(set(baseline) - set(candidate)),
        "candidate_only_keys": sorted(set(candidate) - set(baseline)),
        "allow_subset": allow_subset,
        "expected_count": expected_count,
        "bootstrap_samples": bootstrap_samples,
        "confidence": confidence,
        "results": results,
    }


def _write_markdown(result, output_path):
    lines = [
        f"| Metric | Pairs | Excluded | Base | Candidate | Delta | {result['confidence']:.0%} CI | Improved | FDR p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["results"]:
        lines.append(
            "| {metric} | {n_pairs} | {n_excluded} | {baseline_mean:.4f} | {candidate_mean:.4f} | "
            "{delta_candidate_minus_baseline:+.4f} | "
            "[{delta_ci_low:+.4f}, {delta_ci_high:+.4f}] | "
            "{improved_fraction:.1%} | {wilcoxon_fdr_bh_p:.4g} |".format(**item)
        )
    with open(output_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline metrics_per_utterance.csv")
    parser.add_argument("--candidate", required=True, help="PARR metrics_per_utterance.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--expected-count", type=int, default=3000)
    args = parser.parse_args()

    result = compare(
        args.baseline,
        args.candidate,
        args.bootstrap_samples,
        args.confidence,
        args.seed,
        allow_subset=args.allow_subset,
        expected_count=args.expected_count,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "paired_comparison.json"), "w", encoding="utf-8") as stream:
        json.dump(json_safe(result), stream, indent=2, ensure_ascii=False, allow_nan=False)
    _write_markdown(result, os.path.join(args.output_dir, "paired_comparison.md"))


if __name__ == "__main__":
    main()
