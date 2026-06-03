# AML Repro Guide

This guide is for the local AML copy in `baselines/aml-main_copy`. It is written for direct hands-on use: one environment setup, one entry command shape, and ready-to-run commands for every dataset that is currently wired into this copy.

## 1. What This Copy Supports

The runnable task keys in this copy are:

- `imdb`
- `emotions`
- `sst`
- `agn`
- `rtn`

Alias support is also enabled:

- `emotion` / `emr` -> `emotions`
- `sst2` -> `sst`
- `ag_news` -> `agn`
- `rotten_tomatoes` / `rotten-tomatoes` -> `rtn`

Current limitation:

- `eraser_movie_reviews` is **not** wired into `config/tasks.py` in this AML copy, so it is not part of the ready-to-run command list below.

## 2. Working Directory

Run AML from the AML root, not from the repository root:

```bash
cd /Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy
```

This matters because the project uses relative imports and local default paths.

## 3. Environment Setup

AML's `requirements.txt` is incomplete for this copy. In practice, you should install the listed requirements plus the missing runtime packages imported by the code.

Recommended setup:

```bash
cd /Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy
conda activate aml
pip install -r requirements.txt
pip install optuna peft scikit-learn pandas sentencepiece
```

If your PyTorch / CUDA stack is managed elsewhere, keep that environment and install the missing Python packages into it instead.

## 4. Model / Cache Paths

Two defaults are defined in [config/constants.py](/Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy/config/constants.py):

- `HF_CACHE = "./"`
- `LOCAL_MODELS_PREFIX = "./"`

What this means:

- Hugging Face downloads and caches under the AML working directory by default.
- Local LLaMA / Mistral paths are expected under:
  - `./DOWNLOADED_MODELS/meta-llama_Llama-2-7b-hf`
  - `./DOWNLOADED_MODELS/mistralai_Mistral-7B-v0.1`

If your local model storage lives elsewhere, edit `LOCAL_MODELS_PREFIX` before running LLM experiments.

## 5. Entry Command

AML's main entry is:

```bash
python runs/run.py <task> <explained_backbone> <interpreter_backbone> <metric>
```

Arguments:

- `task`: one of `imdb / emotions / sst / agn / rtn`
- `explained_backbone`: `BERT / ROBERTA / DISTILBERT / LLAMA / MISTRAL`
- `interpreter_backbone`: `BERT / ROBERTA / DISTILBERT`
- `metric`: one of
  - `SUFFICIENCY`
  - `COMPREHENSIVENESS`
  - `EVAL_LOG_ODDS`
  - `AOPC_SUFFICIENCY`
  - `AOPC_COMPREHENSIVENESS`
  - `AOPC_COMPREHENSIVENESS_AOPC_SUFFICIENCY`
  - `COMPREHENSIVENESS_SUFFICIENCY`

Important constraint:

- `interpreter_backbone` cannot be `LLAMA` or `MISTRAL` in this code path. Only the explained model can be an LLM.

## 6. What One Run Actually Does

One AML command triggers the full pipeline in [runs/run.py](/Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy/runs/run.py):

1. Hyper-parameter search
2. Pre-train interpreter
3. Inference on the pre-trained interpreter
4. Instance-wise fine-tune

So a single run is not a lightweight inference call; it is a multi-stage training pipeline.

## 7. Ready-to-Run Commands Per Dataset

### 7.1 Recommended encoder-only starter runs

These are the safest first commands because they use public encoder models rather than local LLM checkpoints.

```bash
cd /Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy
conda activate aml
```

IMDB:

```bash
python runs/run.py imdb BERT BERT SUFFICIENCY
```

Emotion:

```bash
python runs/run.py emotions BERT BERT SUFFICIENCY
```

SST-2:

```bash
python runs/run.py sst BERT BERT SUFFICIENCY
```

AG News:

```bash
python runs/run.py agn BERT BERT SUFFICIENCY
```

Rotten Tomatoes:

```bash
python runs/run.py rtn BERT BERT SUFFICIENCY
```

### 7.2 If you want ROBERTA instead

Swap both backbone arguments:

```bash
python runs/run.py imdb ROBERTA ROBERTA SUFFICIENCY
```

The same pattern applies to `emotions / sst / agn / rtn`.

### 7.3 If you want an LLM explained model

Use a local LLaMA or Mistral checkpoint for the explained model, and keep an encoder interpreter:

```bash
python runs/run.py emotions LLAMA ROBERTA SUFFICIENCY
```

```bash
python runs/run.py imdb MISTRAL ROBERTA SUFFICIENCY
```

Before these run successfully, make sure the local model paths expected by `config/tasks.py` actually exist.

## 8. Outputs

AML writes to `OUT/` by default, as configured in [config/config.py](/Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy/config/config.py).

Key directories:

- `OUT/CONFIG`
- `OUT/PRE_TRAIN`
- `OUT/INFERENCE_PRETRAIN`
- `OUT/FINE_TUNE`
- `OUT/RUNNING_TIMES`

Useful artifacts to check after a run:

- Optuna result pickle under `OUT/CONFIG/OPTUNA_RESULTS`
- Pre-train checkpoints under `OUT/PRE_TRAIN/CHECKPOINTS`
- Fine-tune CSV results under `OUT/FINE_TUNE/RESULTS_DF`

## 9. Dataset-Specific Notes

### IMDB

- Long texts
- The task config truncates LLM tokenizers more aggressively (`llm_explained_tokenizer_max_length = 400`)
- Good first target if you want to probe long-text AML behavior

### Emotion

- Short text, multi-class
- Has LoRA-related local path fields in the task config for LLM variants
- Easiest place to sanity-check multi-class behavior

### SST-2

- Short binary sentiment
- Good smoke-test dataset because examples are short and public encoder checkpoints are easy to fetch

### AG News

- Multi-class news classification
- Larger label space than sentiment tasks

### Rotten Tomatoes

- Short binary sentiment
- Similar operational profile to SST-2, but with a different text distribution

## 10. Common Failure Points

### `ModuleNotFoundError` or import issues

Make sure you are running from:

```bash
cd /Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy
```

and then calling:

```bash
python runs/run.py ...
```

### Missing LLaMA / Mistral checkpoints

Encoder-only commands do not need local LLM checkpoints.

LLM commands do.

If you see local model path errors, either:

1. switch to encoder-only runs first, or
2. edit `LOCAL_MODELS_PREFIX` in `config/constants.py`

### Missing Python packages

`requirements.txt` alone is not enough for this copy. If a run fails on imports, install:

```bash
python -m pip install optuna peft scikit-learn pandas sentencepiece
```

### Movie Reviews / ERASER

The helper [movie_reviews.py](/Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy/movie_reviews.py) converts ERASER-style raw files, but this task is not currently exposed through `config/tasks.py` or `runs/run.py`. Treat it as a separate extension task, not a ready-made entry.

## 11. Suggested Repro Order

To minimize friction:

1. `sst` with `BERT BERT SUFFICIENCY`
2. `rtn` with `BERT BERT SUFFICIENCY`
3. `emotions` with `BERT BERT SUFFICIENCY`
4. `imdb` with `BERT BERT SUFFICIENCY`
5. only then try `LLAMA` or `MISTRAL`

That order gives you a short-text smoke test first, then longer texts, then local-LLM dependency checks.
