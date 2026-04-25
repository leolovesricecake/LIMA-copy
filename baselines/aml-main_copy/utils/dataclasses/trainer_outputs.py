from dataclasses import dataclass
from typing import Any, Union

from torch import Tensor

from utils.dataclasses.loss_output import LossOutput


@dataclass
class AmlOutput:
    loss_output: LossOutput
    explained_model_predicted_class: Tensor
    explained_model_predicted_logits: Tensor
    tokens_attr: Tensor


@dataclass
class StepOutput:
    tokens_attr: Tensor
    input: Any
    explained_model_predicted_class: Tensor
    explained_model_logits: Tensor

    prediction_loss: Union[Tensor, None] = None
    prediction_loss_weighted: Union[Tensor, None] = None
    regularization_loss: Union[Tensor, None] = None
    regularization_weighted: Union[Tensor, None] = None
    inverse_loss: Union[Tensor, None] = None
    inverse_loss_weighted: Union[Tensor, None] = None

    loss: Union[Tensor, None] = None

