import copy
from typing import Tuple, Union

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
        self.special_tokens = special_tokens
        self.perturbation_steps = torch.arange(10, 100, 10)
        self.device = get_device()

        self.labels_tokens = None

    def log_odds(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.get_indices(item_args)
        prob_original = torch.softmax(item_args.explained_model_predicted_logits, dim = 0)

        inputs = copy.deepcopy(item_args.input)

        inputs.input_ids[0][topk_indices] = self.ref_token_id

        inputs_ids, attention_mask = merge_prompts(  #
            inputs = inputs.input_ids, attention_mask = inputs.attention_mask,
            task_prompt = inputs.task_prompt_input_ids,
            label_prompt = inputs.label_prompt_input_ids,
            task_prompt_attention_mask = inputs.task_prompt_attention_mask,
            label_prompt_attention_mask = inputs.label_prompt_attention_mask  #
        )
        logits_perturbed = run_model(model = self.model, model_backbone = ExpArgs.explained_model_backbone,
                                     input_ids = inputs_ids.cuda(), attention_mask = attention_mask.cuda(),
                                     is_return_logits = True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim = 0)
        result = (torch.log(prob_perturbed[item_args.explained_model_predicted_class]) - torch.log(
            prob_original[item_args.explained_model_predicted_class])).item()

        return result

    def sufficiency(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.get_indices(item_args)
        prob_original = torch.softmax(item_args.explained_model_predicted_logits, dim = 0)

        # if len(topk_indices) == 0:
        if topk_indices.shape[-1] == 0:
            # topk% is too less to select even word - so no masking will happen.
            return 0

        inputs = copy.deepcopy(item_args.input)
        mask = torch.zeros_like(inputs.input_ids[0]).bool()
        mask[topk_indices] = 1
        if required_tokens is not None:
            mask[required_tokens] = 1
        masked_input_ids = inputs.input_ids[0][mask].unsqueeze(0)
        masked_attention_mask = inputs.attention_mask[0][mask].unsqueeze(0)

        # print(f"EVAL - before merge: {masked_input_ids}")
        masked_input_ids, masked_attention_mask = merge_prompts(  #
            inputs = masked_input_ids, attention_mask = masked_attention_mask,
            task_prompt = inputs.task_prompt_input_ids,
            label_prompt = inputs.label_prompt_input_ids,
            task_prompt_attention_mask = inputs.task_prompt_attention_mask,
            label_prompt_attention_mask = inputs.label_prompt_attention_mask  #
        )
        # print(f"EVAL - before merge: {masked_input_ids}")

        logits_perturbed = run_model(model = self.model, model_backbone = ExpArgs.explained_model_backbone,
                                     input_ids = masked_input_ids.cuda(), attention_mask = masked_attention_mask.cuda(),
                                     is_return_logits = True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim = 0)

        result = (prob_original[item_args.explained_model_predicted_class] - prob_perturbed[item_args.explained_model_predicted_class]).item()
        return result

    def comprehensiveness(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.get_indices(item_args)
        prob_original = torch.softmax(item_args.explained_model_predicted_logits, dim = 0)

        inputs = copy.deepcopy(item_args.input)
        mask = torch.ones_like(inputs.input_ids[0]).bool()
        mask[topk_indices] = 0
        if required_tokens is not None:
            mask[required_tokens] = 1

        masked_input_ids = inputs.input_ids[0][mask].unsqueeze(0)
        masked_attention_mask = inputs.attention_mask[0][mask].unsqueeze(0)

        masked_input_ids, masked_attention_mask = merge_prompts(inputs = masked_input_ids,
                                                                attention_mask = masked_attention_mask,
                                                                task_prompt = inputs.task_prompt_input_ids,
                                                                label_prompt = inputs.label_prompt_input_ids,
                                                                task_prompt_attention_mask = inputs.task_prompt_attention_mask,
                                                                label_prompt_attention_mask = inputs.label_prompt_attention_mask)
        logits_perturbed = run_model(model = self.model, model_backbone = ExpArgs.explained_model_backbone,
                                     input_ids = masked_input_ids.cuda(), attention_mask = masked_attention_mask.cuda(),
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
        input_ids: Tensor = item_args.input.input_ids
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            input_ids = input_ids.squeeze()
        else:
            input_ids = input_ids[0]
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
