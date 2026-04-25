import torch
from torch import nn

from config.config import ExpArgs
from config.types_enums import LabelTokenPosition, ActivationFunctionTypes


def add_label_handler(last_hidden_states, inserted_label_token_indices):
    interpreter_label_token_position = ExpArgs.interpreter_label_token_position

    if interpreter_label_token_position == LabelTokenPosition.NONE.value:
        return last_hidden_states

    if interpreter_label_token_position == LabelTokenPosition.FIRST_TOKEN.value:
        return last_hidden_states[:, 1:]

    elif interpreter_label_token_position == LabelTokenPosition.LAST_TOKEN.value:
        return last_hidden_states[:, :-1]

    elif interpreter_label_token_position == LabelTokenPosition.AFTER_LAST_SEP.value:
        return torch.stack(
            [torch.cat((row[:idx], row[idx + 1:])) for row, idx in zip(last_hidden_states, inserted_label_token_indices)])

    else:
        raise ValueError(f"post_model_pre_classifier ERROR!")


def post_model_pre_classifier(last_hidden_states, inserted_label_token_indices):
    last_hidden_states = add_label_handler(last_hidden_states, inserted_label_token_indices)
    return last_hidden_states


def get_activation_layer():
    activation_function = ExpArgs.interpreter_classifier_activation_function
    if activation_function == ActivationFunctionTypes.RELU.value:
        return nn.ReLU()
    elif activation_function == ActivationFunctionTypes.TANH.value:
        return nn.Tanh()
    else:
        raise ValueError("Activation function not supported")


def create_interpreter_classifier(hidden_size):
    if ExpArgs.interpreter_classifier_size == 2:
        return nn.Sequential(

            nn.Linear(in_features = hidden_size, out_features = hidden_size),

            get_activation_layer(),

            nn.Linear(hidden_size, 1)

        )

    elif ExpArgs.interpreter_classifier_size == 1:
        return nn.Sequential(nn.Linear(hidden_size, 1))

    else:
        raise ValueError("Unsupported create_interpreter_classifier")
