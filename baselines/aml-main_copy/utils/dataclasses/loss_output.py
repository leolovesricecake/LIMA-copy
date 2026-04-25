from dataclasses import dataclass

from torch import Tensor


@dataclass
class LossOutput:
    loss: Tensor
    prediction_loss_weighted: Tensor
    inverse_loss_weighted: Tensor
    regularization_loss_weighted: Tensor
    prediction_loss: Tensor
    inverse_loss: Tensor
    regularization_loss: Tensor

