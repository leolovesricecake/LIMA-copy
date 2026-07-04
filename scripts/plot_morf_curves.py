from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TRAJECTORY_SUMMARY_CSV = "trajectory_summary.csv"
TRAJECTORY_POINTS_CSV = "trajectory_points.csv"
TRAJECTORY_POINTS_JSONL = "trajectory_points.jsonl"
CURVE_SUMMARY_CSV = "curve_summary.csv"

SUMMARY_GROUP_COLUMNS = [
    "source_family",
    "run_id",
    "report_stage",
    "dataset",
    "split",
    "model_name",
    "method_name",
    "step_index",
    "total_steps",
    "delete_count",
    "delete_fraction",
    "remaining_fraction",
]

DATASET_ALIASES = {
    "sst": "sst2",
    "sst2": "sst2",
    "rtn": "rotten_tomatoes",
    "rotten-tomatoes": "rotten_tomatoes",
    "rotten_tomatoes": "rotten_tomatoes",
    "emotions": "emotion",
    "emotion": "emotion",
    "eraser": "eraser_movie_reviews",
    "eraser-movie-reviews": "eraser_movie_reviews",
    "eraser_movie_reviews": "eraser_movie_reviews",
    "imdb": "imdb",
}


def _slugify(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "unknown"
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-_.").lower()
    return normalized or "unknown"


def _canonical_dataset_name(name: str) -> str:
    key = str(name or "").strip().lower()
    return DATASET_ALIASES.get(key, _slugify(key))


def _normalize_model_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "unknown-model"
    candidate = Path(raw.rstrip("/")).name or raw
    candidate = candidate.removeprefix("model-")
    return _slugify(candidate)


def _normalize_method_name(name: str) -> str:
    return _slugify(str(name or "unknown-method"))


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding = "utf-8"))


def _load_eval_report(run_dir: Path) -> Dict:
    report_path = run_dir / "eval_report.json"
    if report_path.exists():
        return _read_json(report_path)
    return {}


def _fill_metadata_from_report(frame: pd.DataFrame, run_dir: Path, fallback_source_family: str) -> pd.DataFrame:
    frame = frame.copy()
    report = _load_eval_report(run_dir)
    report_method = report.get("report_method")
    dataset = report.get("dataset")
    split = report.get("split")
    model_name = report.get("model_name")
    report_stage = report.get("report_stage")

    if "source_family" not in frame.columns:
        inferred = fallback_source_family
        if str(report_method).strip().lower() == "aml":
            inferred = "aml"
        frame["source_family"] = inferred
    if "run_id" not in frame.columns:
        frame["run_id"] = str(report.get("run_id") or run_dir.name)
    if "report_stage" not in frame.columns:
        frame["report_stage"] = str(report_stage or ("EVAL" if fallback_source_family == "lima_llm" else "UNKNOWN"))
    if "dataset" not in frame.columns:
        frame["dataset"] = str(dataset or run_dir.parent.parent.name)
    if "split" not in frame.columns:
        frame["split"] = str(split or "")
    if "model_name" not in frame.columns:
        fallback_model = run_dir.parent.name if run_dir.parent.name.startswith("model-") else run_dir.parent.name
        frame["model_name"] = str(model_name or fallback_model)
    if "method_name" not in frame.columns:
        frame["method_name"] = str(report_method or run_dir.name)
    return frame


def _aggregate_points(points: pd.DataFrame) -> pd.DataFrame:
    grouped = points.groupby(SUMMARY_GROUP_COLUMNS, dropna = False)
    summary = grouped.agg(
        mean_target_probability = ("target_probability", "mean"),
        std_target_probability = ("target_probability", "std"),
        mean_prob_drop_from_full = ("prob_drop_from_full", "mean"),
        std_prob_drop_from_full = ("prob_drop_from_full", "std"),
        sample_count = ("sample_id", pd.Series.nunique),
    ).reset_index()
    summary["std_target_probability"] = summary["std_target_probability"].fillna(0.0)
    summary["std_prob_drop_from_full"] = summary["std_prob_drop_from_full"].fillna(0.0)
    return summary


def _load_points(path: Path, fallback_source_family: str) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding = "utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path)
        if "deleted_ids" in frame.columns:
            frame["deleted_ids"] = frame["deleted_ids"].fillna("[]")
    frame = _fill_metadata_from_report(frame, path.parent, fallback_source_family)
    return _aggregate_points(frame)


def _load_summary(path: Path, fallback_source_family: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _fill_metadata_from_report(frame, path.parent, fallback_source_family)
    return frame


def _load_curve_summary(path: Path, fallback_source_family: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _fill_metadata_from_report(frame, path.parent, fallback_source_family)
    return frame


def _scan_roots(roots: Sequence[str], fallback_source_family: str) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    seen_run_dirs = set()
    for root in roots:
        root_path = Path(root)
        for curve_path in sorted(root_path.rglob(CURVE_SUMMARY_CSV)):
            seen_run_dirs.add(curve_path.parent.resolve())
            frames.append(_load_curve_summary(curve_path, fallback_source_family))

        for summary_path in sorted(root_path.rglob(TRAJECTORY_SUMMARY_CSV)):
            if summary_path.parent.resolve() in seen_run_dirs:
                continue
            seen_run_dirs.add(summary_path.parent.resolve())
            frames.append(_load_summary(summary_path, fallback_source_family))

        for points_name in (TRAJECTORY_POINTS_CSV, TRAJECTORY_POINTS_JSONL):
            for points_path in sorted(root_path.rglob(points_name)):
                if points_path.parent.resolve() in seen_run_dirs:
                    continue
                seen_run_dirs.add(points_path.parent.resolve())
                frames.append(_load_points(points_path, fallback_source_family))
    return frames


def _prepare_frame(aml_roots: Sequence[str], lima_roots: Sequence[str]) -> pd.DataFrame:
    frames = []
    frames.extend(_scan_roots(aml_roots, "aml"))
    frames.extend(_scan_roots(lima_roots, "lima_llm"))
    if not frames:
        raise SystemExit("No curve_summary.csv, trajectory_summary.csv, or trajectory_points files were found.")

    frame = pd.concat(frames, ignore_index = True)
    frame["dataset"] = frame["dataset"].map(_canonical_dataset_name)
    frame["model_name"] = frame["model_name"].map(_normalize_model_name)
    frame["method_name"] = frame["method_name"].map(_normalize_method_name)
    frame["report_stage"] = frame["report_stage"].fillna("").map(lambda value: str(value).strip())
    frame["run_id"] = frame["run_id"].fillna("").map(lambda value: str(value).strip())
    if "curve_type" not in frame.columns:
        frame["curve_type"] = "deletion"
    frame["curve_type"] = frame["curve_type"].fillna("deletion").map(lambda value: str(value).strip().lower())
    if "mean_prob_delta_from_full" not in frame.columns and "mean_prob_drop_from_full" in frame.columns:
        frame["mean_prob_delta_from_full"] = frame["mean_prob_drop_from_full"]
    if "std_prob_delta_from_full" not in frame.columns and "std_prob_drop_from_full" in frame.columns:
        frame["std_prob_delta_from_full"] = frame["std_prob_drop_from_full"]
    if "mean_prob_drop_from_full" not in frame.columns and "mean_prob_delta_from_full" in frame.columns:
        frame["mean_prob_drop_from_full"] = frame["mean_prob_delta_from_full"]
    if "std_prob_drop_from_full" not in frame.columns and "std_prob_delta_from_full" in frame.columns:
        frame["std_prob_drop_from_full"] = frame["std_prob_delta_from_full"]
    return frame


def _apply_filters(frame: pd.DataFrame, datasets, models, methods) -> pd.DataFrame:
    filtered = frame
    if datasets:
        allowed = {_canonical_dataset_name(item) for item in datasets}
        filtered = filtered[filtered["dataset"].isin(allowed)]
    if models:
        allowed = {_normalize_model_name(item) for item in models}
        filtered = filtered[filtered["model_name"].isin(allowed)]
    if methods:
        allowed = {_normalize_method_name(item) for item in methods}
        filtered = filtered[filtered["method_name"].isin(allowed)]
    return filtered


def _pooled_stats(group: pd.DataFrame, mean_col: str, std_col: str) -> Dict[str, float]:
    counts = group["sample_count"].fillna(0).astype(float)
    means = group[mean_col].astype(float)
    stds = group[std_col].fillna(0.0).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return {"mean": 0.0, "std": 0.0, "sample_count": 0}

    pooled_mean = float((counts * means).sum() / total)
    pooled_var = float((counts * ((stds ** 2) + ((means - pooled_mean) ** 2))).sum() / total)
    return {"mean": pooled_mean, "std": math.sqrt(max(pooled_var, 0.0)), "sample_count": int(total)}


def _line_label_map(group: pd.DataFrame) -> Dict[int, str]:
    label_map: Dict[int, str] = {}
    stage_counts = group.groupby("method_name")["report_stage"].nunique(dropna = True).to_dict()
    for idx, row in group.iterrows():
        method_name = str(row["method_name"])
        report_stage = str(row["report_stage"]).strip().lower()
        if stage_counts.get(method_name, 0) > 1 and report_stage:
            label_map[idx] = f"{method_name}@{report_stage}"
        else:
            label_map[idx] = method_name
    return label_map


def _plot_group(
        group: pd.DataFrame,
        *,
        output_dir: Path,
        y_mode: str,
        x_col: str,
        title_template: str,
        dpi: int) -> List[Dict[str, object]]:
    group = group.copy()
    group["line_label"] = pd.Series(_line_label_map(group))

    mean_col = "mean_target_probability" if y_mode == "probability" else "mean_prob_delta_from_full"
    std_col = "std_target_probability" if y_mode == "probability" else "std_prob_delta_from_full"
    ylabel = "Target Probability" if y_mode == "probability" else "Probability Drop From Full"
    xlabel = x_col.replace("_", " ").title()

    dataset = str(group["dataset"].iloc[0])
    model_name = str(group["model_name"].iloc[0])
    output_path = output_dir / f"{model_name}-{dataset}.png"

    fig, ax = plt.subplots(figsize = (7.5, 4.8))
    manifest_rows: List[Dict[str, object]] = []

    for line_label, line_group in sorted(group.groupby("line_label"), key = lambda item: item[0]):
        pooled_rows = []
        for _, step_group in line_group.groupby([x_col], dropna = False):
            stats = _pooled_stats(step_group, mean_col = mean_col, std_col = std_col)
            first_row = step_group.iloc[0]
            pooled_rows.append(
                {
                    "x": float(first_row[x_col]),
                    "mean": float(stats["mean"]),
                    "std": float(stats["std"]),
                }
            )

        pooled_frame = pd.DataFrame(pooled_rows).sort_values(["x"])
        ax.plot(
            pooled_frame["x"].to_numpy(),
            pooled_frame["mean"].to_numpy(),
            linewidth = 2.0,
            label = line_label,
        )
        if len(pooled_frame) > 1 and float(pooled_frame["std"].max()) > 0.0:
            ax.fill_between(
                pooled_frame["x"].to_numpy(),
                (pooled_frame["mean"] - pooled_frame["std"]).to_numpy(),
                (pooled_frame["mean"] + pooled_frame["std"]).to_numpy(),
                alpha = 0.18,
            )

        run_sample_counts = (
            line_group.groupby(["run_id", "report_stage"], dropna = False)["sample_count"].max().reset_index()
        )
        manifest_rows.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "method_name": line_label,
                "report_stages": ",".join(sorted({str(value) for value in line_group["report_stage"].tolist() if str(value)})),
                "run_ids": ",".join(sorted({str(value) for value in line_group["run_id"].tolist() if str(value)})),
                "sample_count": int(run_sample_counts["sample_count"].sum()) if not run_sample_counts.empty else 0,
                "curve_type": ",".join(sorted({str(value) for value in line_group["curve_type"].tolist() if str(value)})),
                "x_axis": x_col,
                "step_count": int(pooled_frame["x"].nunique()),
                "output_png": str(output_path),
            }
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title_template.format(model_name = model_name, dataset = dataset))
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha = 0.25, linewidth = 0.6)
    ax.legend(frameon = False)
    fig.tight_layout()
    fig.savefig(output_path, dpi = dpi)
    plt.close(fig)

    return manifest_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Plot perturbation curves from AML and lima_llm curve artifacts.")
    parser.add_argument("--aml-root", action = "append", default = [], help = "AML result root. Can be passed multiple times.")
    parser.add_argument("--lima-root", action = "append", default = [], help = "lima_llm result root. Can be passed multiple times.")
    parser.add_argument("--output-dir", required = True, help = "Directory to save <model_name>-<dataset>.png outputs.")
    parser.add_argument("--y", choices = ["probability", "drop"], default = "probability", help = "Y-axis metric.")
    parser.add_argument("--curve", choices = ["deletion", "retention"], default = "deletion", help = "Curve family to plot.")
    parser.add_argument(
        "--x",
        choices = ["auto", "delete_fraction", "remaining_fraction", "keep_fraction"],
        default = "auto",
        help = "X-axis field. auto uses keep_fraction for retention and delete_fraction for deletion.",
    )
    parser.add_argument("--dataset", action = "append", default = [], help = "Optional dataset filter. Can be repeated.")
    parser.add_argument("--model", action = "append", default = [], help = "Optional model filter. Can be repeated.")
    parser.add_argument("--method", action = "append", default = [], help = "Optional method filter. Can be repeated.")
    parser.add_argument("--title-template", default = "{model_name} | {dataset}", help = "Plot title template.")
    parser.add_argument("--dpi", type = int, default = 200, help = "PNG DPI.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.aml_root and not args.lima_root:
        raise SystemExit("At least one --aml-root or --lima-root must be provided.")

    frame = _prepare_frame(args.aml_root, args.lima_root)
    frame = _apply_filters(frame, args.dataset, args.model, args.method)
    frame = frame[frame["curve_type"] == str(args.curve)]
    if frame.empty:
        raise SystemExit("No curve data remained after applying filters.")
    x_col = str(args.x)
    if x_col == "auto":
        x_col = "keep_fraction" if str(args.curve) == "retention" else "delete_fraction"
    if x_col not in frame.columns:
        raise SystemExit(f"Requested x-axis column is unavailable: {x_col}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    manifest_rows: List[Dict[str, object]] = []
    for _, group in frame.groupby(["dataset", "model_name"], dropna = False):
        manifest_rows.extend(
            _plot_group(
                group,
                output_dir = output_dir,
                y_mode = args.y,
                x_col = x_col,
                title_template = args.title_template,
                dpi = args.dpi,
            )
        )

    manifest_path = output_dir / "curve_plot_manifest.csv"
    pd.DataFrame(manifest_rows).sort_values(["dataset", "model_name", "method_name"]).to_csv(
        manifest_path,
        index = False,
        encoding = "utf-8-sig",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
