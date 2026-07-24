# AML Repro Guide

This guide is for the local AML copy in `baselines/aml-main_copy`.

## 1. Canonical Tasks

The mainline task names are aligned with the shared `mobius` package:

- `sst2`
- `eraser_movie_reviews`
- `imdb`
- `rotten_tomatoes`
- `emotion`

Compatibility aliases:

- `sst` -> `sst2`
- `rtn`, `rotten-tomatoes` -> `rotten_tomatoes`
- `emotions` -> `emotion`
- `agn`, `ag_news` remain available as AML-only legacy tasks

Important:

- `emotion` is the 6-class `dair-ai/emotion` dataset
- `eraser_movie_reviews` is the binary ERASER movie-reviews dataset
- `emr` is rejected as ambiguous; use one of the canonical names above

## 2. Working Directory

```bash
cd baselines/aml-main_copy
```

## 3. Environment Setup

```bash
conda activate aml
pip install -r requirements.txt
pip install optuna peft scikit-learn pandas sentencepiece
```

AML now reuses `mobius.data.load_dataset_bundle` for the five canonical tasks, so make sure the repo root stays available when running from this directory.

## 4. Model and Cache Paths

Defaults live in [config/constants.py](/Users/apple/LEO/codes/LIMA-main/baselines/aml-main_copy/config/constants.py):

- `HF_CACHE = "/mnt/huawei/nsq/temp/hf"`
- `LOCAL_MODELS_PREFIX = "/mnt/huawei/nsq/models"`

Adjust those if your training machine stores local models elsewhere.

## 5. Main AML Command

```bash
python runs/run.py <task> <explained_backbone> <interpreter_backbone> <metric>
```

Optional dataset-source arguments shared with `mobius`:

```bash
--eraser-root /path/or/hf://dataset
--sst2-source /path/or/hf://dataset
--dataset-cache-dir /path/to/cache
```

Optional explained-model override:

```bash
--explained_model_path /path/to/model
```

Arguments:

- `task`: one of `sst2 / eraser_movie_reviews / imdb / rotten_tomatoes / emotion`
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

Notes:

- Only the explained model can be an LLM in this code path.
- The five canonical tasks all support prompt-based decoder-only explained models.
- Binary tasks use label verbalizers `N / P`.
- `emotion` uses `A / B / C / D / E / F`.
- AML validates those verbalizers up front and fails early if the tokenizer maps any of them to multiple tokens.

## 6. What One AML Run Does

One `runs/run.py` command triggers:

1. Hyper-parameter search
2. Interpreter pre-train
3. Pretrained-interpreter inference
4. Instance-wise fine-tune

## 7. Ready-to-Run Examples

Encoder-only starter runs:

```bash
python runs/run.py sst2 BERT BERT SUFFICIENCY
python runs/run.py imdb BERT BERT SUFFICIENCY
python runs/run.py rotten_tomatoes BERT BERT SUFFICIENCY
python runs/run.py emotion BERT BERT SUFFICIENCY
python runs/run.py eraser_movie_reviews BERT BERT SUFFICIENCY --explained_model_path /path/to/eraser_classifier
```

LLM explained-model runs:

```bash
CUDA_VISIBLE_DEVICES=0 python runs/run.py sst2 LLAMA ROBERTA AOPC_COMPREHENSIVENESS --explained_model_path /mnt/huawei/nsq/models/meta-llama/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=5 python runs/run.py rotten_tomatoes LLAMA ROBERTA SUFFICIENCY --explained_model_path /mnt/huawei/nsq/models/meta-llama/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=5 python runs/run.py emotion LLAMA ROBERTA SUFFICIENCY --explained_model_path /mnt/huawei/nsq/models/meta-llama/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=7 python runs/run.py eraser_movie_reviews LLAMA ROBERTA SUFFICIENCY --eraser-root hf://eraser-benchmark/movie_rationales --explained_model_path /mnt/huawei/nsq/models/meta-llama/Llama-3.1-8B-Instruct

CUDA_VISIBLE_DEVICES= python runs/run.py imdb LLAMA ROBERTA SUFFICIENCY --explained_model_path /mnt/huawei/nsq/models/Qwen/Qwen2.5-7B-Instruct
```

If you use a local or alternate SST-2 source:

```bash
python runs/run.py sst2 BERT BERT SUFFICIENCY --sst2-source hf://nyu-mll/glue
```

If you use a local or alternate ERASER source:

```bash
python runs/run.py eraser_movie_reviews BERT BERT SUFFICIENCY \
  --eraser-root hf://eraser-benchmark/movie_rationales \
  --explained_model_path /path/to/eraser_classifier
```

## 8. Training an Encoder Explained Model

`eraser_movie_reviews` does not ship with a default encoder classifier checkpoint in this AML copy.

Train one with:

```bash
python runs/train_explained_model.py eraser_movie_reviews BERT \
  --output-dir OUT/TRAINED_EXPLAINED_MODELS/eraser_bert \
  --eraser-root hf://eraser-benchmark/movie_rationales
```

The same script supports `BERT / ROBERTA / DISTILBERT` for all five canonical tasks:

```bash
python runs/train_explained_model.py emotion ROBERTA \
  --output-dir OUT/TRAINED_EXPLAINED_MODELS/emotion_roberta
```

After training, point AML to the saved checkpoint:

```bash
python runs/run.py eraser_movie_reviews BERT BERT SUFFICIENCY \
  --explained_model_path OUT/TRAINED_EXPLAINED_MODELS/eraser_bert \
  --eraser-root hf://eraser-benchmark/movie_rationales
```

Training outputs include:

- `aml_explained_model_metrics.json`
- `aml_explained_model_report.json`

The report captures the canonical task name, dataset source arguments, label metadata, training hyperparameters, final evaluation metrics, and saved artifact paths.

## 9. Outputs

AML writes to `OUT/` by default:

- `OUT/CONFIG`
- `OUT/PRE_TRAIN`
- `OUT/INFERENCE_PRETRAIN`
- `OUT/FINE_TUNE`
- `OUT/RUNNING_TIMES`

Important result artifacts:

- `results.csv`
- `all_metrics_results_long.csv`
- `all_metrics_results_wide.csv`
- `all_metrics_summary.csv`
- `all_metrics_report.json`

## 10. Legacy Notes

- `agn` is still runnable, but it is outside the mainline five-dataset alignment target.
- Some historical documents and experiments in this folder still refer to `emotions`, `sst`, or `rtn`; those are compatibility names only.
