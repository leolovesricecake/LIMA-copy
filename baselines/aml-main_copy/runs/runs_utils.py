from __future__ import annotations

from config.tasks import (
    AGN_TASK,
    EMOTION_TASK,
    ERASER_MOVIE_REVIEWS_TASK,
    IMDB_TASK,
    ROTTEN_TOMATOES_TASK,
    SST2_TASK,
)


_TASKS = {
    "agn": AGN_TASK,
    "ag_news": AGN_TASK,
    "emotion": EMOTION_TASK,
    "emotions": EMOTION_TASK,
    "eraser_movie_reviews": ERASER_MOVIE_REVIEWS_TASK,
    "eraser-movie-reviews": ERASER_MOVIE_REVIEWS_TASK,
    "eraser": ERASER_MOVIE_REVIEWS_TASK,
    "imdb": IMDB_TASK,
    "rotten_tomatoes": ROTTEN_TOMATOES_TASK,
    "rotten-tomatoes": ROTTEN_TOMATOES_TASK,
    "rtn": ROTTEN_TOMATOES_TASK,
    "sst": SST2_TASK,
    "sst2": SST2_TASK,
}


def get_task(raw_name: str):
    key = str(raw_name).strip().lower()
    if key == "emr":
        raise ValueError(
            "AML task name 'emr' is ambiguous. Use 'emotion' for the 6-class Hugging Face emotion dataset "
            "or 'eraser_movie_reviews' for the ERASER movie reviews benchmark."
        )
    if key not in _TASKS:
        supported = ", ".join(sorted(_TASKS.keys()))
        raise ValueError(f"Unsupported AML task '{raw_name}'. Supported task keys: {supported}")
    return _TASKS[key]
