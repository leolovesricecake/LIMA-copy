"""Backward-compatible entry point for the standalone Sparse Mobius runner."""

try:
    from mobius_verify.scripts.run_sparse_mobius_llm import main
except ModuleNotFoundError:
    from run_sparse_mobius_llm import main


if __name__ == "__main__":
    main()
