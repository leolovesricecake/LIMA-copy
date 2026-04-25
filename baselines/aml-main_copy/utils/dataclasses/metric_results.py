from dataclasses import dataclass
from typing import Union


@dataclass
class MetricResults:
    epoch: int
    step: int
    item_index: str
    task: str
    evaluation_metric: str
    explained_model_backbone: str
    interpreter_model_backbone: str
    metric_result: float
    metric_result_str: str
    metric_steps_result: any
    steps_k: any
    explained_model_predicted_class: Union[int, None]
    token_evaluation_option: str
