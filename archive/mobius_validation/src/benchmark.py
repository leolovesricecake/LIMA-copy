"""Compatibility imports for the standalone Sparse Mobius runner.

The old controlled/native orchestrator was intentionally removed.
"""

from .sparse_runner import explain_sample, run_sparse_mobius

__all__ = ["explain_sample", "run_sparse_mobius"]
