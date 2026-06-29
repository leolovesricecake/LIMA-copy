import copy
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from transformers import AutoTokenizer

from config.config import ExpArgs
from config.types_enums import EvalTokens
from utils.dataclasses.evaluations import DataForEvaluation
from utils.utils_functions import get_device, run_model, merge_prompts, is_model_encoder_only


class MetricsFunctions:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id, special_tokens: Tensor):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.perturbation_steps = torch.arange(10, 100, 10)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device(get_device())

        self.special_tokens = special_tokens.to(self.device)
        self.labels_tokens = None

    @staticmethod
    def _to_int(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.squeeze().item())
        return int(value)

    def _get_target_label(self, item_args: DataForEvaluation) -> int:
        return self._to_int(item_args.explained_model_predicted_class)

    def _get_sequence_inputs(self, item_args: DataForEvaluation) -> Tuple[Tensor, Tensor]:
        input_ids = self._get_single_item_tensor(item_args.input.input_ids).to(self.device)
        attention_mask = self._get_single_item_tensor(item_args.input.attention_mask).to(self.device)
        return input_ids, attention_mask

    def _predict_probability(
            self,
            item_args: DataForEvaluation,
            masked_input_ids: Tensor,
            masked_attention_mask: Tensor,
            target_label: int) -> float:
        masked_input_ids, masked_attention_mask = merge_prompts(
            inputs = masked_input_ids,
            attention_mask = masked_attention_mask,
            task_prompt = item_args.input.task_prompt_input_ids,
            label_prompt = item_args.input.label_prompt_input_ids,
            task_prompt_attention_mask = item_args.input.task_prompt_attention_mask,
            label_prompt_attention_mask = item_args.input.label_prompt_attention_mask)
        logits_perturbed = run_model(model = self.model,
                                     model_backbone = ExpArgs.explained_model_backbone,
                                     input_ids = masked_input_ids.to(self.device),
                                     attention_mask = masked_attention_mask.to(self.device),
                                     is_return_logits = True).squeeze()
        probabilities = torch.softmax(logits_perturbed, dim = 0)
        return float(probabilities[target_label].item())

    def get_ranked_deletion_indices(self, item_args: DataForEvaluation) -> Tuple[Tensor, Union[Tensor, None]]:
        tokens_attr, _, required_tokens = self.eval_tokens_handler(item_args)
        ranked_indices = torch.argsort(tokens_attr, descending = True)
        if required_tokens is None:
            return ranked_indices, required_tokens

        required_tokens = required_tokens.to(ranked_indices.device)
        eligible_mask = torch.ones_like(tokens_attr, dtype = torch.bool)
        eligible_mask[required_tokens] = False
        ranked_indices = ranked_indices[eligible_mask[ranked_indices]]
        return ranked_indices, required_tokens

    def deletion_trajectory(self, item_args: DataForEvaluation) -> Dict[str, Any]:
        ranked_indices, required_tokens = self.get_ranked_deletion_indices(item_args)
        target_label = self._get_target_label(item_args)
        prob_original = torch.softmax(item_args.explained_model_predicted_logits.to(self.device), dim = 0)
        p_full = float(prob_original[target_label].item())
        input_ids, attention_mask = self._get_sequence_inputs(item_args)

        total_steps = int(ranked_indices.shape[0])
        points: List[Dict[str, Any]] = [
            dict(
                step_index = 0,
                total_steps = total_steps,
                delete_count = 0,
                delete_fraction = 0.0,
                remaining_fraction = 1.0,
                target_probability = p_full,
                prob_drop_from_full = 0.0,
                is_full_text_step = True,
                deleted_ids = [],
            )
        ]

        for step in range(1, total_steps + 1):
            deleted_indices = ranked_indices[:step]
            mask = torch.ones_like(input_ids, dtype = torch.bool)
            mask[deleted_indices] = False
            if required_tokens is not None:
                mask[required_tokens.to(mask.device)] = True

            masked_input_ids = input_ids[mask].unsqueeze(0)
            masked_attention_mask = attention_mask[mask].unsqueeze(0)
            p_step = self._predict_probability(
                item_args = item_args,
                masked_input_ids = masked_input_ids,
                masked_attention_mask = masked_attention_mask,
                target_label = target_label)

            delete_fraction = float(step / total_steps) if total_steps > 0 else 0.0
            points.append(
                dict(
                    step_index = int(step),
                    total_steps = total_steps,
                    delete_count = int(step),
                    delete_fraction = delete_fraction,
                    remaining_fraction = float(1.0 - delete_fraction),
                    target_probability = p_step,
                    prob_drop_from_full = float(p_full - p_step),
                    is_full_text_step = False,
                    deleted_ids = [int(index) for index in deleted_indices.tolist()],
                )
            )

        aopc = 0.0
        if points:
            aopc = float(sum(float(point["prob_drop_from_full"]) for point in points) / len(points))

        return dict(
            target_label_id = int(target_label),
            full_probability = p_full,
            aopc = aopc,
            points = points,
        )

    def aopc(self, item_args: DataForEvaluation, trajectory_payload: Dict[str, Any] | None = None) -> float:
        payload = trajectory_payload if trajectory_payload is not None else self.deletion_trajectory(item_args)
        return float(payload["aopc"])

    @staticmethod
    def _get_single_item_tensor(value):
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                raise ValueError("evaluation expects sequence tensors, but received a scalar tensor")
            if value.dim() == 1:
                return value
            return value[0]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("evaluation expects batch_size=1 inputs")
            return MetricsFunctions._get_single_item_tensor(value[0])
        raise TypeError(f"unsupported input type for evaluation: {type(value)}")

    def log_odds(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.get_indices(item_args)
        prob_original = torch.softmax(item_args.explained_model_predicted_logits.to(self.device), dim = 0)

        inputs = copy.deepcopy(item_args.input)
        sequence_input_ids = self._get_single_item_tensor(inputs.input_ids)
        sequence_input_ids[topk_indices.to(sequence_input_ids.device)] = self.ref_token_id

        inputs_ids, attention_mask = merge_prompts(  #
            inputs = inputs.input_ids, attention_mask = inputs.attention_mask,
            task_prompt = inputs.task_prompt_input_ids,
            label_prompt = inputs.label_prompt_input_ids,
            task_prompt_attention_mask = inputs.task_prompt_attention_mask,
            label_prompt_attention_mask = inputs.label_prompt_attention_mask  #
        )
        logits_perturbed = run_model(model = self.model, model_backbone = ExpArgs.explained_model_backbone,
                                     input_ids = inputs_ids.to(self.device), attention_mask = attention_mask.to(self.device),
                                     is_return_logits = True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim = 0)
        result = (torch.log(prob_perturbed[item_args.explained_model_predicted_class]) - torch.log(
            prob_original[item_args.explained_model_predicted_class])).item()

        return result

    def sufficiency(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.get_indices(item_args)

        device = next(self.model.parameters()).device

        prob_original = torch.softmax(
            item_args.explained_model_predicted_logits.to(device),
            dim=0
        )

        if topk_indices.shape[-1] == 0:
            return 0

        inputs = copy.deepcopy(item_args.input)
        # print('\n- ', inputs.input_ids)
        # print('- ', inputs.attention_mask, '\n')

        input_ids = self._get_single_item_tensor(inputs.input_ids).to(device)
        attention_mask = self._get_single_item_tensor(inputs.attention_mask).to(device)

        mask = torch.zeros_like(input_ids).bool()
        mask[topk_indices.to(device)] = 1

        if required_tokens is not None:
            mask[required_tokens.to(device)] = 1

        masked_input_ids = input_ids[mask].unsqueeze(0)
        masked_attention_mask = attention_mask[mask].unsqueeze(0)

        masked_input_ids, masked_attention_mask = merge_prompts(
            inputs=masked_input_ids,
            attention_mask=masked_attention_mask,
            task_prompt=inputs.task_prompt_input_ids,
            label_prompt=inputs.label_prompt_input_ids,
            task_prompt_attention_mask=inputs.task_prompt_attention_mask,
            label_prompt_attention_mask=inputs.label_prompt_attention_mask
        )

        logits_perturbed = run_model(
            model=self.model,
            model_backbone=ExpArgs.explained_model_backbone,
            input_ids=masked_input_ids.to(device),
            attention_mask=masked_attention_mask.to(device),
            is_return_logits=True
        ).squeeze()

        prob_perturbed = torch.softmax(logits_perturbed, dim=0)

        cls = item_args.explained_model_predicted_class
        result = (prob_original[cls] - prob_perturbed[cls]).item()

        return result

    def comprehensiveness(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.get_indices(item_args)
        prob_original = torch.softmax(item_args.explained_model_predicted_logits.to(self.device), dim = 0)

        inputs = copy.deepcopy(item_args.input)
        input_ids = self._get_single_item_tensor(inputs.input_ids)
        attention_mask = self._get_single_item_tensor(inputs.attention_mask)
        mask = torch.ones_like(input_ids).bool()
        mask[topk_indices.to(mask.device)] = 0
        if required_tokens is not None:
            mask[required_tokens.to(mask.device)] = 1

        masked_input_ids = input_ids[mask].unsqueeze(0)
        masked_attention_mask = attention_mask[mask].unsqueeze(0)

        masked_input_ids, masked_attention_mask = merge_prompts(inputs = masked_input_ids,
                                                                attention_mask = masked_attention_mask,
                                                                task_prompt = inputs.task_prompt_input_ids,
                                                                label_prompt = inputs.label_prompt_input_ids,
                                                                task_prompt_attention_mask = inputs.task_prompt_attention_mask,
                                                                label_prompt_attention_mask = inputs.label_prompt_attention_mask)
        logits_perturbed = run_model(model = self.model, model_backbone = ExpArgs.explained_model_backbone,
                                     input_ids = masked_input_ids.to(self.device),
                                     attention_mask = masked_attention_mask.to(self.device),
                                     is_return_logits = True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim = 0)

        result = (prob_original[item_args.explained_model_predicted_class] - prob_perturbed[item_args.explained_model_predicted_class]).item()
        return result

    def get_indices(self, item_args: DataForEvaluation) -> Tuple[Tensor, Tensor]:
        tokens_attr, n_attr, required_tokens = self.eval_tokens_handler(item_args)
        k = int(n_attr * item_args.k / 100)
        topk_indices = torch.topk(tokens_attr, k, sorted = False).indices

        if required_tokens is not None:
            overlap = bool(set(topk_indices.tolist()).intersection(set(required_tokens.tolist())))
            if overlap:
                raise ValueError(f"required_tokens souled not be in the topk_indices")
        return topk_indices, required_tokens

    def eval_tokens_handler(self, item_args: DataForEvaluation) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
        val = float('-inf')
        tokens_attr: Tensor = copy.deepcopy(item_args.tokens_attr)
        input_ids = self._get_single_item_tensor(item_args.input.input_ids).to(self.device)
        n_attr = tokens_attr.shape[-1]
        required_tokens = None

        # not encoder only support EvalTokens.ALL_TOKENS.value only
        # if ExpArgs.eval_tokens == EvalTokens.ALL_TOKENS.value:
        #     return tokens_attr, n_attr, required_tokens
        # elif ExpArgs.eval_tokens == EvalTokens.NO_CLS.value:
        #     if is_model_encoder_only():
        #         tokens_attr[0] = val  # cls
        #         n_attr = n_attr - 1  # cls
        #         required_tokens = torch.tensor([0])  # cls
        #     else:
        #         raise ValueError("unsupported EvalTokens. NO_CLS.value for not encoders only models")
        #     return tokens_attr, n_attr, required_tokens
        if ExpArgs.eval_tokens == EvalTokens.NO_SPECIAL_TOKENS.value:
            indices = torch.isin(input_ids, self.special_tokens)
            if indices.sum() == 0:
                return tokens_attr, n_attr, required_tokens
            tokens_attr[indices] = val
            required_tokens = torch.nonzero(indices).squeeze()
            if required_tokens.dim() == 0:
                required_tokens = required_tokens.unsqueeze(0)
            n_attr = n_attr - required_tokens.shape[-1]
            return tokens_attr, n_attr, required_tokens
        else:
            raise ValueError("unsupported ExpArgs.eval_tokens selected")
