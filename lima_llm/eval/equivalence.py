from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _path_str(path: Sequence[Any]) -> str:
    return ".".join(str(item) for item in path) if path else "<root>"


def _path_is_ignored(path: Sequence[Any], ignored_paths: Sequence[Tuple[Any, ...]]) -> bool:
    for ignored in ignored_paths:
        if len(ignored) <= len(path) and tuple(path[: len(ignored)]) == tuple(ignored):
            return True
    return False


def _component_as_dict(component: Dict[str, Any]) -> Dict[str, float]:
    return {
        "confidence": float(component.get("confidence", 0.0)),
        "effectiveness": float(component.get("effectiveness", 0.0)),
        "consistency": float(component.get("consistency", 0.0)),
        "collaboration": float(component.get("collaboration", 0.0)),
    }


def compare_trace_dicts(
    left_trace: Sequence[Dict[str, Any]],
    right_trace: Sequence[Dict[str, Any]],
    tolerance: float,
) -> Dict[str, Any]:
    if len(left_trace) != len(right_trace):
        return {
            "passed": False,
            "reason": "trace_length_mismatch",
            "left_length": len(left_trace),
            "right_length": len(right_trace),
            "max_abs_diff": None,
            "first_failure": {"path": "trace", "left": len(left_trace), "right": len(right_trace)},
        }

    max_abs_diff = 0.0
    first_failure = None
    for idx, (left_item, right_item) in enumerate(zip(left_trace, right_trace)):
        if int(left_item.get("step", -1)) != int(right_item.get("step", -1)):
            return {
                "passed": False,
                "reason": "trace_step_mismatch",
                "max_abs_diff": max_abs_diff,
                "first_failure": {
                    "path": f"trace[{idx}].step",
                    "left": left_item.get("step"),
                    "right": right_item.get("step"),
                },
            }
        if int(left_item.get("selected_chunk_id", -1)) != int(right_item.get("selected_chunk_id", -1)):
            return {
                "passed": False,
                "reason": "trace_selected_id_mismatch",
                "max_abs_diff": max_abs_diff,
                "first_failure": {
                    "path": f"trace[{idx}].selected_chunk_id",
                    "left": left_item.get("selected_chunk_id"),
                    "right": right_item.get("selected_chunk_id"),
                },
            }

        scalar_keys = ["marginal_gain", "total_score"]
        for key in scalar_keys:
            left_value = float(left_item.get(key, 0.0))
            right_value = float(right_item.get(key, 0.0))
            diff = abs(left_value - right_value)
            if diff > max_abs_diff:
                max_abs_diff = diff
            if diff > tolerance and first_failure is None:
                first_failure = {
                    "path": f"trace[{idx}].{key}",
                    "left": left_value,
                    "right": right_value,
                    "abs_diff": diff,
                }

        left_components = _component_as_dict(dict(left_item.get("components", {})))
        right_components = _component_as_dict(dict(right_item.get("components", {})))
        for key in ["confidence", "effectiveness", "consistency", "collaboration"]:
            left_value = float(left_components[key])
            right_value = float(right_components[key])
            diff = abs(left_value - right_value)
            if diff > max_abs_diff:
                max_abs_diff = diff
            if diff > tolerance and first_failure is None:
                first_failure = {
                    "path": f"trace[{idx}].components.{key}",
                    "left": left_value,
                    "right": right_value,
                    "abs_diff": diff,
                }

    return {
        "passed": first_failure is None,
        "reason": "ok" if first_failure is None else "trace_numeric_mismatch",
        "max_abs_diff": max_abs_diff,
        "first_failure": first_failure,
    }


def compare_sample_payloads(
    left_payload: Dict[str, Any],
    right_payload: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    left_selected = list(left_payload.get("selected_chunk_ids", []))
    right_selected = list(right_payload.get("selected_chunk_ids", []))
    if left_selected != right_selected:
        return {
            "passed": False,
            "reason": "selected_chunk_ids_mismatch",
            "max_abs_diff": None,
            "first_failure": {
                "path": "selected_chunk_ids",
                "left": left_selected,
                "right": right_selected,
            },
        }

    trace_result = compare_trace_dicts(
        left_trace=list(left_payload.get("trace", [])),
        right_trace=list(right_payload.get("trace", [])),
        tolerance=tolerance,
    )
    if not trace_result["passed"]:
        return trace_result

    return {
        "passed": True,
        "reason": "ok",
        "max_abs_diff": trace_result["max_abs_diff"],
        "first_failure": None,
    }


def compare_json_objects(
    left_obj: Any,
    right_obj: Any,
    tolerance: float,
    ignored_paths: Sequence[Tuple[Any, ...]],
    path: Tuple[Any, ...] = (),
) -> Dict[str, Any]:
    if _path_is_ignored(path, ignored_paths):
        return {"passed": True, "max_abs_diff": 0.0, "first_failure": None}

    if isinstance(left_obj, dict) and isinstance(right_obj, dict):
        left_keys = set(left_obj.keys())
        right_keys = set(right_obj.keys())
        if left_keys != right_keys:
            return {
                "passed": False,
                "max_abs_diff": None,
                "first_failure": {
                    "path": _path_str(path),
                    "left_keys_only": sorted(left_keys - right_keys),
                    "right_keys_only": sorted(right_keys - left_keys),
                },
            }

        max_abs_diff = 0.0
        first_failure = None
        for key in sorted(left_keys):
            child = compare_json_objects(
                left_obj=left_obj[key],
                right_obj=right_obj[key],
                tolerance=tolerance,
                ignored_paths=ignored_paths,
                path=(*path, key),
            )
            if child["max_abs_diff"] is not None:
                max_abs_diff = max(max_abs_diff, float(child["max_abs_diff"]))
            if not child["passed"]:
                first_failure = child["first_failure"]
                break
        return {"passed": first_failure is None, "max_abs_diff": max_abs_diff, "first_failure": first_failure}

    if isinstance(left_obj, list) and isinstance(right_obj, list):
        if len(left_obj) != len(right_obj):
            return {
                "passed": False,
                "max_abs_diff": None,
                "first_failure": {
                    "path": _path_str(path),
                    "left_length": len(left_obj),
                    "right_length": len(right_obj),
                },
            }

        max_abs_diff = 0.0
        first_failure = None
        for idx, (left_item, right_item) in enumerate(zip(left_obj, right_obj)):
            child = compare_json_objects(
                left_obj=left_item,
                right_obj=right_item,
                tolerance=tolerance,
                ignored_paths=ignored_paths,
                path=(*path, idx),
            )
            if child["max_abs_diff"] is not None:
                max_abs_diff = max(max_abs_diff, float(child["max_abs_diff"]))
            if not child["passed"]:
                first_failure = child["first_failure"]
                break
        return {"passed": first_failure is None, "max_abs_diff": max_abs_diff, "first_failure": first_failure}

    if _is_number(left_obj) and _is_number(right_obj):
        left_value = float(left_obj)
        right_value = float(right_obj)
        diff = abs(left_value - right_value)
        return {
            "passed": diff <= tolerance,
            "max_abs_diff": diff,
            "first_failure": None
            if diff <= tolerance
            else {"path": _path_str(path), "left": left_value, "right": right_value, "abs_diff": diff},
        }

    if left_obj != right_obj:
        return {
            "passed": False,
            "max_abs_diff": None,
            "first_failure": {"path": _path_str(path), "left": left_obj, "right": right_obj},
        }

    return {"passed": True, "max_abs_diff": 0.0, "first_failure": None}


def _load_samples(sample_dir: Path) -> Dict[str, Dict[str, Any]]:
    samples: Dict[str, Dict[str, Any]] = {}
    for sample_path in sorted(sample_dir.glob("*.json")):
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        sample_id = str(payload.get("sample_id", sample_path.stem))
        samples[sample_id] = payload
    return samples


def compare_run_dirs(
    reference_run_dir: Path,
    candidate_run_dir: Path,
    tolerance: float = 1e-6,
    ignored_eval_paths: Sequence[Tuple[Any, ...]] = (
        ("metrics_secondary", "runtime_seconds"),
        ("metrics_secondary", "forward_counters_delta"),
    ),
) -> Dict[str, Any]:
    reference_run_dir = Path(reference_run_dir)
    candidate_run_dir = Path(candidate_run_dir)

    reference_samples_dir = reference_run_dir / "samples"
    candidate_samples_dir = candidate_run_dir / "samples"
    if not reference_samples_dir.exists():
        return {
            "passed": False,
            "reason": "missing_reference_samples",
            "first_failure": {"path": str(reference_samples_dir)},
        }
    if not candidate_samples_dir.exists():
        return {
            "passed": False,
            "reason": "missing_candidate_samples",
            "first_failure": {"path": str(candidate_samples_dir)},
        }

    left_samples = _load_samples(reference_samples_dir)
    right_samples = _load_samples(candidate_samples_dir)
    left_ids = set(left_samples.keys())
    right_ids = set(right_samples.keys())
    if left_ids != right_ids:
        return {
            "passed": False,
            "reason": "sample_id_set_mismatch",
            "first_failure": {
                "left_only": sorted(left_ids - right_ids),
                "right_only": sorted(right_ids - left_ids),
            },
        }

    max_trace_abs_diff = 0.0
    sample_failures: List[Dict[str, Any]] = []
    for sample_id in sorted(left_ids):
        sample_result = compare_sample_payloads(
            left_payload=left_samples[sample_id],
            right_payload=right_samples[sample_id],
            tolerance=tolerance,
        )
        if sample_result["max_abs_diff"] is not None:
            max_trace_abs_diff = max(max_trace_abs_diff, float(sample_result["max_abs_diff"]))
        if not sample_result["passed"]:
            sample_failures.append(
                {
                    "sample_id": sample_id,
                    "reason": sample_result["reason"],
                    "first_failure": sample_result["first_failure"],
                }
            )
            break

    reference_report_path = reference_run_dir / "eval_report.json"
    candidate_report_path = candidate_run_dir / "eval_report.json"
    if not reference_report_path.exists() or not candidate_report_path.exists():
        return {
            "passed": False,
            "reason": "missing_eval_report",
            "first_failure": {
                "reference_exists": reference_report_path.exists(),
                "candidate_exists": candidate_report_path.exists(),
            },
            "sample_check": {
                "passed": len(sample_failures) == 0,
                "max_trace_abs_diff": max_trace_abs_diff,
                "first_failure": sample_failures[0] if sample_failures else None,
            },
        }

    left_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    right_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    report_result = compare_json_objects(
        left_obj=left_report,
        right_obj=right_report,
        tolerance=tolerance,
        ignored_paths=ignored_eval_paths,
        path=(),
    )

    passed = (len(sample_failures) == 0) and bool(report_result["passed"])
    return {
        "passed": passed,
        "reason": "ok" if passed else "run_dir_mismatch",
        "sample_check": {
            "passed": len(sample_failures) == 0,
            "max_trace_abs_diff": max_trace_abs_diff,
            "first_failure": sample_failures[0] if sample_failures else None,
            "sample_count": len(left_ids),
        },
        "report_check": {
            "passed": bool(report_result["passed"]),
            "max_abs_diff": report_result["max_abs_diff"],
            "first_failure": report_result["first_failure"],
            "ignored_paths": [list(path) for path in ignored_eval_paths],
        },
    }

