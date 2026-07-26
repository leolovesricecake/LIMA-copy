"""Schema-v2 run layout, resume state, and atomic result writing."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .artifacts import write_observation_artifact, write_surrogate_artifact
from .config import scientific_config
from .runtime import atomic_write_json, ensure_dir, environment_summary, git_commit
from .schema import AttributionResult, SCHEMA_VERSION


def canonical_digest(payload: object) -> str:
    """Hash a JSON-compatible object with a canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_id(config: Mapping[str, Any]) -> str:
    """Build a compact run ID with a configured suffix or hash fallback."""

    scientific = scientific_config(config)
    budget = int(scientific.get("budget", 0))
    order = int(scientific.get("max_degree", scientific.get("max_order", 0)))
    seed = int(scientific.get("seed", 0))
    configured_suffix = config.get("run_suffix")
    if configured_suffix is None or not str(configured_suffix).strip():
        suffix = canonical_digest(scientific)[:8]
    else:
        suffix = str(configured_suffix).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", suffix):
            raise ValueError(
                "run_suffix must be 1-64 characters using only letters, "
                "digits, '.', '_' or '-', and must start with a letter or digit."
            )
    return f"b{budget}-o{order}-s{seed}-{suffix}"


def model_slug(model_config: Mapping[str, Any]) -> str:
    """Create a filesystem-safe model identifier."""

    raw = str(model_config.get("model_path", model_config.get("type", "unknown")))
    name = raw.rstrip("/").split("/")[-1] or "unknown"
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def default_run_dir(results_dir: str | Path, config: Mapping[str, Any]) -> Path:
    """Resolve the canonical v2 directory for a run."""

    dataset = str(dict(config.get("dataset", {})).get("name", "dataset"))
    method = str(config.get("method", "method"))
    model = model_slug(dict(config.get("model", {})))
    return Path(results_dir) / dataset / model / method / build_run_id(config)


class ResultStore:
    """Own all writes and resume checks for one schema-v2 run."""

    def __init__(
        self,
        run_dir: str | Path,
        config: Mapping[str, Any],
        *,
        output_level: str = "standard",
        command: str | None = None,
        overwrite: bool = False,
        required_artifacts: Sequence[str] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize directories and validate any existing run."""

        self.run_dir = ensure_dir(run_dir)
        self.samples_dir = ensure_dir(self.run_dir / "samples")
        self.failures_dir = ensure_dir(self.run_dir / "failures")
        self.output_level = str(output_level)
        self.required_artifacts = frozenset(str(value) for value in required_artifacts)
        unknown_artifacts = self.required_artifacts - {"observation", "surrogate"}
        if unknown_artifacts:
            raise ValueError(f"Unsupported required artifacts: {sorted(unknown_artifacts)}")
        self.provenance = dict(provenance or {})
        self.observations_dir = (
            ensure_dir(self.run_dir / "observations")
            if "observation" in self.required_artifacts
            else self.run_dir / "observations"
        )
        self.surrogates_dir = (
            ensure_dir(self.run_dir / "surrogates")
            if "surrogate" in self.required_artifacts
            else self.run_dir / "surrogates"
        )
        self.analyses_dir = ensure_dir(self.run_dir / "analyses")
        self.diagnostics_dir = (
            ensure_dir(self.run_dir / "diagnostics")
            if self.output_level == "debug"
            else self.run_dir / "diagnostics"
        )
        self.config = dict(config)
        self.run_id = build_run_id(config)
        self.started = time.time()
        self.completed_ids: list[str] = []
        self.failures: list[Dict[str, Any]] = []
        self.skipped: list[Dict[str, Any]] = []
        self._initialize_run(command=command, overwrite=overwrite)

    def _initialize_run(self, *, command: str | None, overwrite: bool) -> None:
        """Write immutable run metadata and recover completed sample IDs."""

        run_path = self.run_dir / "run.json"
        scientific = scientific_config(self.config)
        fingerprint = canonical_digest(scientific)
        if run_path.exists():
            existing = json.loads(run_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != fingerprint and not overwrite:
                raise ValueError(
                    "Run directory belongs to a different scientific configuration."
                )
        if overwrite:
            self._clear_payload_files()
        if overwrite or not run_path.exists():
            run_payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "run_suffix": self.config.get("run_suffix"),
                "config_fingerprint": fingerprint,
                "scientific_config": scientific,
                "command": command,
                "git_commit": git_commit(self.run_dir),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "environment": environment_summary(),
                "provenance": dict(self.provenance),
            }
            atomic_write_json(run_path, run_payload)
        else:
            self.completed_ids = sorted(
                path.stem
                for path in self.samples_dir.glob("*.json")
                if self.sample_complete(path.stem)
            )
            status_path = self.run_dir / "status.json"
            if status_path.is_file():
                previous = json.loads(status_path.read_text(encoding="utf-8"))
                self.failures = list(previous.get("failures", []))
                self.skipped = list(previous.get("skipped", []))
        self.write_status("running")

    def _clear_payload_files(self) -> None:
        """Clear prior mutable artifacts while preserving the run directory itself."""

        for directory in (
            self.samples_dir,
            self.failures_dir,
            self.diagnostics_dir,
            self.observations_dir,
            self.surrogates_dir,
            self.analyses_dir,
        ):
            if directory.is_dir():
                for path in directory.glob("*"):
                    if path.is_file():
                        path.unlink()
        for filename in ("status.json", "metrics.json"):
            path = self.run_dir / filename
            if path.is_file():
                path.unlink()
        for path in self.run_dir.glob("curves-*.jsonl"):
            path.unlink()

    def sample_complete(self, sample_id: str) -> bool:
        """Return whether a complete sample file already exists."""

        path = self.samples_dir / f"{sample_id}.json"
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        valid = (
            payload.get("schema_version") == SCHEMA_VERSION
            and str(payload.get("sample_id")) == str(sample_id)
            and isinstance(payload.get("ranking"), list)
        )
        if not valid:
            return False
        if "observation" in self.required_artifacts:
            valid = valid and (self.observations_dir / f"{sample_id}.npz").is_file()
        if "surrogate" in self.required_artifacts:
            valid = valid and (self.surrogates_dir / f"{sample_id}.json").is_file()
        return bool(valid)

    def write_sample(self, result: AttributionResult) -> None:
        """Write one explanation and optional debug diagnostics."""

        missing = []
        if "observation" in self.required_artifacts and result.observation_artifact is None:
            missing.append("observation")
        if "surrogate" in self.required_artifacts and result.surrogate_artifact is None:
            missing.append("surrogate")
        if missing:
            raise ValueError(
                f"Sample {result.sample_id} is missing required artifacts: {missing}"
            )
        observation_metadata = None
        if result.observation_artifact is not None:
            observation_metadata = write_observation_artifact(
                self.observations_dir / f"{result.sample_id}.npz",
                result.observation_artifact,
            )
        if result.surrogate_artifact is not None:
            surrogate_payload = dict(result.surrogate_artifact)
            if observation_metadata is not None:
                surrogate_payload.setdefault(
                    "observation_file",
                    f"../observations/{result.sample_id}.npz",
                )
                surrogate_payload.setdefault(
                    "observation_digest",
                    observation_metadata["digest"],
                )
            write_surrogate_artifact(
                self.surrogates_dir / f"{result.sample_id}.json",
                surrogate_payload,
            )
        atomic_write_json(
            self.samples_dir / f"{result.sample_id}.json",
            result.to_dict(self.output_level),
        )
        if self.output_level == "debug" and result.diagnostics:
            atomic_write_json(
                self.diagnostics_dir / f"{result.sample_id}.json",
                result.diagnostics,
            )
        if str(result.sample_id) not in self.completed_ids:
            self.completed_ids.append(str(result.sample_id))

    def record_failure(
        self,
        sample_id: str,
        error: Exception,
        *,
        skipped: bool = False,
    ) -> None:
        """Persist a failed or deliberately excluded sample."""

        payload = {
            "sample_id": str(sample_id),
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
            "skipped": bool(skipped),
        }
        target = self.skipped if skipped else self.failures
        target.append(payload)
        atomic_write_json(self.failures_dir / f"{sample_id}.json", payload)

    def write_status(self, state: str, selected_count: int | None = None) -> None:
        """Refresh the sole run-level resume and completion manifest."""

        atomic_write_json(
            self.run_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "state": str(state),
                "selected_count": selected_count,
                "completed_count": len(self.completed_ids),
                "failed_count": len(self.failures),
                "skipped_count": len(self.skipped),
                "completed_ids": sorted(self.completed_ids),
                "failures": list(self.failures),
                "skipped": list(self.skipped),
                "elapsed_seconds": float(time.time() - self.started),
            },
        )

    def finish(self, selected_count: int) -> Dict[str, Any]:
        """Mark the run complete and return the final status payload."""

        state = "complete" if not self.failures else "complete_with_failures"
        self.write_status(state, selected_count=selected_count)
        return json.loads((self.run_dir / "status.json").read_text(encoding="utf-8"))
