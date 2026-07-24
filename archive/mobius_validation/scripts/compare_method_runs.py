from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mobius_verify.src.statistics import paired_difference_summary
from mobius_verify.src.utils import atomic_write_json, atomic_write_text, ensure_dir


FAITHFULNESS_METRICS = (
    "aopc",
    "comprehensiveness",
    "sufficiency",
    "aopc_comprehensiveness",
    "aopc_sufficiency",
    "log_odds",
)
QUERY_METRICS = (
    "logical_attribution_queries",
    "logical_unique_attribution_queries",
    "unique_attribution_texts",
    "physical_values_scored",
    "model_forward_calls",
    "batch_calls",
    "batch_rows",
    "elapsed_seconds",
)
CONTRACT_FIELDS = (
    "dataset",
    "split",
    "model_path",
    "max_length",
    "prompt_template",
    "chunker",
    "adaptive_profile",
    "adaptive_overrides",
    "mask_operator",
    "empty_perturbation_text",
    "target_mode",
    "value_function",
    "eval_granularity",
    "eval_q_values",
    "verbalizers",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two completed attribution runs without invoking the LLM."
    )
    parser.add_argument("--left-run", required=True)
    parser.add_argument("--right-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", choices=["predicted", "gold"], default="predicted")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--allow-contract-mismatch", action="store_true")
    return parser


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run evaluation with the updated common evaluator first."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _contract(run_root: Path) -> Dict[str, Any]:
    config_path = run_root / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    config = _read_json(config_path)
    contract = config.get("comparison_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Run has no comparison_contract: {config_path}. Re-run it with the new runner."
        )
    return dict(contract)


def _contract_mismatches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    mismatches = []
    for field in CONTRACT_FIELDS:
        if left.get(field) != right.get(field):
            mismatches.append(
                {"field": field, "left": left.get(field), "right": right.get(field)}
            )
    return mismatches


def _sample_payloads(run_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(payload["sample_id"]): payload
        for path in sorted((run_root / "samples").glob("*.json"))
        for payload in [_read_json(path)]
    }


def _eval_rows(run_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in _read_jsonl(run_root / "eval_sample_metrics.jsonl")
    }


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else None


def _write_csv(path: Path, rows: list[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(
    *,
    left_root: Path,
    right_root: Path,
    output_dir: Path,
    target: str = "predicted",
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 2000,
    allow_contract_mismatch: bool = False,
) -> Dict[str, Any]:
    left_contract = _contract(left_root)
    right_contract = _contract(right_root)
    mismatches = _contract_mismatches(left_contract, right_contract)
    if mismatches and not allow_contract_mismatch:
        detail = ", ".join(str(item["field"]) for item in mismatches)
        raise ValueError(f"Comparison contract mismatch in: {detail}")

    left_samples = _sample_payloads(left_root)
    right_samples = _sample_payloads(right_root)
    left_eval = _eval_rows(left_root)
    right_eval = _eval_rows(right_root)
    common_ids = sorted(set(left_samples) & set(right_samples) & set(left_eval) & set(right_eval))
    if not common_ids:
        raise ValueError("The two runs have no common evaluated sample IDs")

    incompatible_samples = []
    compatible_ids = []
    for sample_id in common_ids:
        left_sample = left_samples[sample_id]
        right_sample = right_samples[sample_id]
        same_text = left_sample.get("text") == right_sample.get("text")
        left_chunks = [
            (item.get("start_char"), item.get("end_char"), item.get("text"))
            for item in left_sample.get("chunks", [])
        ]
        right_chunks = [
            (item.get("start_char"), item.get("end_char"), item.get("text"))
            for item in right_sample.get("chunks", [])
        ]
        if not same_text or left_chunks != right_chunks:
            incompatible_samples.append(sample_id)
        else:
            compatible_ids.append(sample_id)
    if incompatible_samples and not allow_contract_mismatch:
        raise ValueError(
            f"Player/text contract differs for {len(incompatible_samples)} samples; "
            f"first={incompatible_samples[0]}"
        )
    paired_ids = compatible_ids
    if not paired_ids:
        raise ValueError("No samples remain after text/chunk contract validation")

    left_method = str(next(iter(left_samples.values())).get("explain_method", "left"))
    right_method = str(next(iter(right_samples.values())).get("explain_method", "right"))
    paired_rows = []
    summaries: Dict[str, Any] = {}

    for metric in FAITHFULNESS_METRICS:
        left_values = []
        right_values = []
        for sample_id in paired_ids:
            left_primary = (
                left_eval[sample_id]
                .get("metrics_by_target", {})
                .get(target, {})
                .get("metrics_primary", {})
            )
            right_primary = (
                right_eval[sample_id]
                .get("metrics_by_target", {})
                .get(target, {})
                .get("metrics_primary", {})
            )
            left_value = left_primary.get(metric)
            right_value = right_primary.get(metric)
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value, (int, float)
            ):
                continue
            left_values.append(float(left_value))
            right_values.append(float(right_value))
            paired_rows.append(
                {
                    "sample_id": sample_id,
                    "metric_family": "faithfulness",
                    "metric": metric,
                    "left": float(left_value),
                    "right": float(right_value),
                    "left_minus_right": float(left_value) - float(right_value),
                }
            )
        summaries[metric] = {
            "left_mean": _mean(left_values),
            "right_mean": _mean(right_values),
            "paired_difference": paired_difference_summary(
                left_values,
                right_values,
                seed=int(bootstrap_seed),
                n_bootstrap=int(bootstrap_samples),
            ),
            "direction": "lower_is_better"
            if metric in {"sufficiency", "aopc_sufficiency", "log_odds"}
            else "higher_is_better",
        }

    query_summaries: Dict[str, Any] = {}
    for metric in QUERY_METRICS:
        left_values = []
        right_values = []
        for sample_id in paired_ids:
            left_value = left_samples[sample_id].get("metadata", {}).get(
                "query_accounting", {}
            ).get(metric)
            right_value = right_samples[sample_id].get("metadata", {}).get(
                "query_accounting", {}
            ).get(metric)
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value, (int, float)
            ):
                continue
            left_values.append(float(left_value))
            right_values.append(float(right_value))
            paired_rows.append(
                {
                    "sample_id": sample_id,
                    "metric_family": "query_cost",
                    "metric": metric,
                    "left": float(left_value),
                    "right": float(right_value),
                    "left_minus_right": float(left_value) - float(right_value),
                }
            )
        query_summaries[metric] = {
            "left_mean": _mean(left_values),
            "right_mean": _mean(right_values),
            "paired_difference": paired_difference_summary(
                left_values,
                right_values,
                seed=int(bootstrap_seed),
                n_bootstrap=int(bootstrap_samples),
            ),
            "direction": "lower_is_better",
        }

    report = {
        "left_run": str(left_root),
        "right_run": str(right_root),
        "left_method": left_method,
        "right_method": right_method,
        "target": str(target),
        "contract_mismatches": mismatches,
        "sample_coverage": {
            "left_explanations": int(len(left_samples)),
            "right_explanations": int(len(right_samples)),
            "common_before_contract_check": int(len(common_ids)),
            "paired_sample_count": int(len(paired_ids)),
            "incompatible_sample_ids": incompatible_samples,
            "left_only_sample_ids": sorted(set(left_samples) - set(right_samples)),
            "right_only_sample_ids": sorted(set(right_samples) - set(left_samples)),
        },
        "faithfulness": summaries,
        "query_cost": query_summaries,
    }
    ensure_dir(output_dir)
    atomic_write_json(output_dir / "comparison_report.json", report)
    _write_csv(output_dir / "paired_sample_metrics.csv", paired_rows)
    lines = [
        "# 归因方法比较",
        "",
        f"- 左侧：{left_method}",
        f"- 右侧：{right_method}",
        f"- 配对样本数：{len(paired_ids)}",
        f"- 评估 target：{target}",
        f"- 契约不一致项：{len(mismatches)}",
        "",
        "## Faithfulness",
        "",
    ]
    for metric, summary in summaries.items():
        delta = summary["paired_difference"]
        lines.append(
            f"- {metric}：left={summary['left_mean']}, right={summary['right_mean']}, "
            f"差值={delta.get('mean_diff')}, 95% CI=[{delta.get('low')}, {delta.get('high')}], "
            f"{summary['direction']}"
        )
    lines.extend(["", "## 查询与时间", ""])
    for metric, summary in query_summaries.items():
        lines.append(
            f"- {metric}：left={summary['left_mean']}, right={summary['right_mean']}"
        )
    atomic_write_text(output_dir / "comparison_report.md", "\n".join(lines) + "\n")
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = compare_runs(
        left_root=Path(args.left_run).expanduser().resolve(),
        right_root=Path(args.right_run).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        target=args.target,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        allow_contract_mismatch=bool(args.allow_contract_mismatch),
    )
    print(
        f"[complete] paired={report['sample_coverage']['paired_sample_count']} "
        f"report={Path(args.output_dir).expanduser().resolve() / 'comparison_report.json'}"
    )


if __name__ == "__main__":
    main()
