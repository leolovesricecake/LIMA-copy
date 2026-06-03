from __future__ import annotations

from config.tasks import AGN_TASK, EMOTION_TASK, IMDB_TASK, RTN_TASK, SST_TASK


_TASKS = {
    "agn": AGN_TASK,
    "ag_news": AGN_TASK,
    "emotion": EMOTION_TASK,
    "emotions": EMOTION_TASK,
    "emr": EMOTION_TASK,
    "imdb": IMDB_TASK,
    "rotten_tomatoes": RTN_TASK,
    "rotten-tomatoes": RTN_TASK,
    "rtn": RTN_TASK,
    "sst": SST_TASK,
    "sst2": SST_TASK,
}


def get_task(raw_name: str):
    key = str(raw_name).strip().lower()
    if key not in _TASKS:
        supported = ", ".join(sorted(_TASKS.keys()))
        raise ValueError(f"Unsupported AML task '{raw_name}'. Supported task keys: {supported}")
    return _TASKS[key]
