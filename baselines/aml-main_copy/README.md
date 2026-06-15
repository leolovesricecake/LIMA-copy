# Implementation of Attributive Masking Learning (AML)

<p align="center">
  <img width="1400" src="examples.png" alt="AML" title="AML">
</p>

## Introduction

This directory contains a local AML copy wired to the repository's mainline text-classification datasets.

Canonical AML task names are now aligned with `lima_llm`:

- `sst2`
- `eraser_movie_reviews`
- `imdb`
- `rotten_tomatoes`
- `emotion`

Legacy aliases remain available for compatibility:

- `sst` -> `sst2`
- `rtn`, `rotten-tomatoes` -> `rotten_tomatoes`
- `emotions` -> `emotion`
- `agn`, `ag_news` remain available as AML-only legacy tasks

Important naming clarification:

- `emotion` is the 6-class Hugging Face `dair-ai/emotion` dataset
- `eraser_movie_reviews` is the binary ERASER movie-reviews benchmark
- `emr` is no longer accepted because it is ambiguous between those two datasets

## Running AML

The main entry point is:

```bash
cd baselines/aml-main_copy
python runs/run.py <task> <explained_backbone> <interpreter_backbone> <metric>
```

Extra dataset-source flags are available for the shared `lima_llm` loader:

```bash
python runs/run.py eraser_movie_reviews BERT BERT SUFFICIENCY --eraser-root hf://eraser-benchmark/movie_rationales
python runs/run.py sst2 BERT BERT SUFFICIENCY --sst2-source hf://nyu-mll/glue
```

You can still override the explained model path directly:

```bash
python runs/run.py imdb LLAMA ROBERTA SUFFICIENCY --explained_model_path /path/to/local/model
```

Prompt-based decoder-only AML is supported for all five canonical datasets:

- binary datasets use `N / P`
- `emotion` uses `A / B / C / D / E / F`

AML now validates these verbalizers up front and fails early if the tokenizer does not map them to single tokens.

## Training an Encoder Explained Model

`eraser_movie_reviews` does not assume a baked-in encoder classifier checkpoint. Train one with:

```bash
cd baselines/aml-main_copy
python runs/train_explained_model.py eraser_movie_reviews BERT --output-dir OUT/TRAINED_EXPLAINED_MODELS/eraser_bert
```

The output is a standard Hugging Face checkpoint and can be plugged back into AML:

```bash
python runs/run.py eraser_movie_reviews BERT BERT SUFFICIENCY --explained_model_path OUT/TRAINED_EXPLAINED_MODELS/eraser_bert
```

The same training script also supports `sst2 / imdb / rotten_tomatoes / emotion` and `BERT / ROBERTA / DISTILBERT`.
Each training run now saves both:

- `aml_explained_model_metrics.json`
- `aml_explained_model_report.json`

The report file includes task identity, dataset-source arguments, training hyperparameters, label metadata, final metrics, and artifact paths.

## More Details

For a fuller setup guide, task notes, and example commands, see `REPRO_GUIDE.md`.
