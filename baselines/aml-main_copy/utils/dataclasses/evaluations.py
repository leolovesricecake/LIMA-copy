from dataclasses import dataclass
from typing import Optional, Sequence, Union

from torch import Tensor


@dataclass
class DataForEvaluationInputs:
    input_ids: Union[Tensor, Sequence[Tensor]]
    attention_mask: Union[Tensor, Sequence[Tensor]]
    task_prompt_input_ids: Optional[Union[Tensor, Sequence[Tensor]]]
    label_prompt_input_ids: Optional[Tensor]
    task_prompt_attention_mask: Optional[Union[Tensor, Sequence[Tensor]]]
    label_prompt_attention_mask: Optional[Tensor]


@dataclass
class DataForEvaluation:
    tokens_attr: Tensor
    input: DataForEvaluationInputs
    explained_model_predicted_class: Tensor
    explained_model_predicted_logits: Tensor
    k: float = 0.0
