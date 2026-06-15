import gc
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from peft import PeftModel
from torch import Tensor, nn
from transformers import (BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AutoTokenizer, AutoConfig,
                          AutoModelForCausalLM)

from config.config import ExpArgs, BackbonesMetaData
from config.constants import HF_CACHE, NEW_ADDED_TRAINABLE_PARAMS
from config.types_enums import ModelBackboneTypes, LabelTokenPosition, RefTokenNameTypes
from models.interpreter_models.bert_interpreter import BertInterpreter
from models.interpreter_models.distilbert_interpreter import DistilBertInterpreter
from models.interpreter_models.roberta_interpreter import RobertaInterpreter
from utils.dataclasses import Task
from utils.utils_functions import get_device, is_model_encoder_only


def is_local_model_path(model_path: str) -> bool:
    return Path(str(model_path)).expanduser().exists()


def get_local_files_only(model_path: str) -> bool:
    return is_local_model_path(model_path)


def normalize_token_id(token_id):
    if isinstance(token_id, (list, tuple)):
        if len(token_id) == 0:
            return None
        return token_id[0]
    return token_id


def get_task_fine_tuned_model_path(task: Task, model_backbone: str) -> Optional[str]:
    if model_backbone == ModelBackboneTypes.BERT.value:
        return task.bert_fine_tuned_model
    if model_backbone == ModelBackboneTypes.ROBERTA.value:
        return task.roberta_fine_tuned_model
    if model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return task.distilbert_fine_tuned_model
    return None


def get_task_base_model_path(task: Task, model_backbone: str) -> Optional[str]:
    if model_backbone == ModelBackboneTypes.BERT.value:
        return task.bert_base_model
    if model_backbone == ModelBackboneTypes.ROBERTA.value:
        return task.roberta_base_model
    if model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return task.distilbert_base_model
    if model_backbone == ModelBackboneTypes.LLAMA.value:
        return task.llama_model
    if model_backbone == ModelBackboneTypes.MISTRAL.value:
        return task.mistral_model
    return None


def get_default_encoder_training_model_path(task: Task, model_backbone: str) -> str:
    model_path = get_task_base_model_path(task, model_backbone)
    if model_path is None:
        raise ValueError(f"Task '{task.name}' does not define a base model for backbone '{model_backbone}'")
    return model_path


def build_prompt_label_vocab_tokens(task: Task, tokenizer, explained_model_path: str) -> torch.Tensor:
    invalid_labels: List[str] = []
    label_vocab_tokens: List[int] = []
    for label_symbol in list(task.labels_int_str_maps.keys()):
        token_ids = tokenizer.encode(str(label_symbol), add_special_tokens = False)
        if len(token_ids) != 1:
            invalid_labels.append(f"{label_symbol} -> {token_ids}")
            continue
        label_vocab_tokens.append(int(token_ids[0]))

    if invalid_labels:
        raise ValueError(
            "Prompt-based AML requires each label verbalizer to map to exactly one tokenizer token. "
            f"task={task.name}, explained_model_path={explained_model_path}, invalid_labels={invalid_labels}"
        )

    return torch.tensor(label_vocab_tokens, dtype = torch.long)


def resolve_explained_model_path(task: Task) -> str:
    if ExpArgs.explained_model_path is not None:
        return ExpArgs.explained_model_path
    if is_model_encoder_only(ExpArgs.explained_model_backbone):
        explained_model_path = get_task_fine_tuned_model_path(task, ExpArgs.explained_model_backbone)
        if explained_model_path is None:
            raise ValueError(
                f"Task '{task.name}' does not define a default {ExpArgs.explained_model_backbone} explained-model checkpoint. "
                "Train one with runs/train_explained_model.py or pass --explained_model_path."
            )
        return explained_model_path
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        return task.llama_model
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.MISTRAL.value:
        return task.mistral_model
    raise ValueError("unsupported model backbone explained model selected")


def ensure_llm_pad_token(tokenizer_or_config):
    pad_token_id = normalize_token_id(getattr(tokenizer_or_config, "pad_token_id", None))
    if pad_token_id is None:
        eos_token_id = normalize_token_id(getattr(tokenizer_or_config, "eos_token_id", None))
        unk_token_id = normalize_token_id(getattr(tokenizer_or_config, "unk_token_id", None))
        if eos_token_id is not None:
            tokenizer_or_config.pad_token_id = eos_token_id
        elif unk_token_id is not None:
            tokenizer_or_config.pad_token_id = unk_token_id
        else:
            raise ValueError("LLM tokenizer/config must expose eos_token_id or unk_token_id to derive pad_token_id")
    return tokenizer_or_config


def load_explained_model():
    task = ExpArgs.task
    explained_model_path = resolve_explained_model_path(task)
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(explained_model_path, cache_dir = HF_CACHE,
                                                              local_files_only = get_local_files_only(explained_model_path))
        model.to(get_device())
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained(explained_model_path, cache_dir = HF_CACHE,
                                                                 local_files_only = get_local_files_only(explained_model_path))
        model.to(get_device())
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(explained_model_path,
                                                                    cache_dir = HF_CACHE,
                                                                    local_files_only = get_local_files_only(explained_model_path))
        model.to(get_device())
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        from transformers import LlamaForCausalLM, LlamaForSequenceClassification
        model_path = explained_model_path
        if task.is_llm_use_lora:
            if ExpArgs.explained_model_path is not None:
                raise ValueError("--explained_model_path is not supported for LoRA-based AML tasks. "
                                 "This task expects task-specific sequence-classification adapters.")
            model = LlamaForSequenceClassification.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                                   cache_dir = HF_CACHE,
                                                                   num_labels = len(task.labels_int_str_maps.keys()),
                                                                   local_files_only = True, device_map = "auto")
            model = PeftModel.from_pretrained(model, task.llama_adapter, device_map = "auto")
            model = model.merge_and_unload()

        else:
            if ExpArgs.explained_model_path is not None:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                             cache_dir = HF_CACHE,
                                                             local_files_only = get_local_files_only(model_path),
                                                             device_map = "auto",
                                                             trust_remote_code = True)
            else:
                model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16, cache_dir = HF_CACHE,
                                                         local_files_only = True, device_map = "auto")

        ensure_llm_pad_token(model.config)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.MISTRAL.value:
        from transformers import MistralForCausalLM, MistralForSequenceClassification
        model_path = explained_model_path
        if task.is_llm_use_lora:
            if ExpArgs.explained_model_path is not None:
                raise ValueError("--explained_model_path is not supported for LoRA-based AML tasks. "
                                 "This task expects task-specific sequence-classification adapters.")
            model = MistralForSequenceClassification.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                                     cache_dir = HF_CACHE,
                                                                     num_labels = len(task.labels_int_str_maps.keys()))
            model = PeftModel.from_pretrained(model, task.mistral_adapter)
            model = model.merge_and_unload()
        else:
            if ExpArgs.explained_model_path is not None:
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                             cache_dir = HF_CACHE,
                                                             local_files_only = get_local_files_only(model_path),
                                                             device_map = "auto",
                                                             trust_remote_code = True)
            else:
                model = MistralForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16,
                                                           cache_dir = HF_CACHE)

        ensure_llm_pad_token(model.config)

    else:
        raise ValueError("unsupported model backbone explained model selected")

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    return model


# For llm model - do not use the fine-tuned interpreter model option
def get_interpreter_model_path(task: Task):
    if ExpArgs.fine_tuned_interpreter_model_path is not None:
        return ExpArgs.fine_tuned_interpreter_model_path
    elif ExpArgs.interpreter_model_backbone == ModelBackboneTypes.BERT.value:
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            return task.bert_fine_tuned_model or task.bert_base_model
        return task.bert_base_model
    elif ExpArgs.interpreter_model_backbone == ModelBackboneTypes.ROBERTA.value:
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            return task.roberta_fine_tuned_model or task.roberta_base_model
        return task.roberta_base_model
    elif ExpArgs.interpreter_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            return task.distilbert_fine_tuned_model or task.distilbert_base_model
        return task.distilbert_base_model
    else:
        raise ValueError("unsupported model backbone selected - interpreter model")


def load_interpreter_model():
    task = ExpArgs.task
    model_path: str = get_interpreter_model_path(task)
    interpreter_model_backbone = ExpArgs.interpreter_model_backbone
    if interpreter_model_backbone == ModelBackboneTypes.BERT.value:
        return BertInterpreter.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif interpreter_model_backbone == ModelBackboneTypes.ROBERTA.value:
        return RobertaInterpreter.from_pretrained(model_path, cache_dir = HF_CACHE)
    elif interpreter_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        return DistilBertInterpreter.from_pretrained(model_path, cache_dir = HF_CACHE)
    else:
        raise ValueError("unsupported model backbone selected")


def get_interpreter_config():
    task = ExpArgs.task
    model_path: str = get_interpreter_model_path(task)
    return AutoConfig.from_pretrained(model_path)


def get_models_tokenizer(model_backbone, is_explained_model: bool = False):
    task = ExpArgs.task
    if model_backbone == ModelBackboneTypes.BERT.value:
        tokenizer_path = resolve_explained_model_path(task) if is_explained_model else get_interpreter_model_path(task)
        return BertTokenizer.from_pretrained(tokenizer_path, cache_dir = HF_CACHE,
                                             local_files_only = get_local_files_only(tokenizer_path))
    elif model_backbone == ModelBackboneTypes.ROBERTA.value:
        tokenizer_path = resolve_explained_model_path(task) if is_explained_model else get_interpreter_model_path(task)
        return RobertaTokenizer.from_pretrained(tokenizer_path, cache_dir = HF_CACHE,
                                                local_files_only = get_local_files_only(tokenizer_path))
    elif model_backbone == ModelBackboneTypes.DISTILBERT.value:
        tokenizer_path = resolve_explained_model_path(task) if is_explained_model else get_interpreter_model_path(task)
        return DistilBertTokenizer.from_pretrained(tokenizer_path, cache_dir = HF_CACHE,
                                                   local_files_only = get_local_files_only(tokenizer_path))
    elif model_backbone in [ModelBackboneTypes.LLAMA.value, ModelBackboneTypes.MISTRAL.value]:
        if is_explained_model:
            tokenizer_path = resolve_explained_model_path(task)
        elif model_backbone == ModelBackboneTypes.LLAMA.value:
            tokenizer_path = task.llama_model
        else:
            tokenizer_path = task.mistral_model
        new_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                      cache_dir = HF_CACHE,
                                                      padding_side = 'left',
                                                      local_files_only = get_local_files_only(tokenizer_path),
                                                      trust_remote_code = True)
        ensure_llm_pad_token(new_tokenizer)
        return new_tokenizer
    else:
        raise ValueError("unsupported model type selected")


def get_warmup_steps_and_total_training_steps(n_epochs: int, train_samples_length: int, batch_size: int,
                                              warmup_ratio: int, accumulate_grad_batches: int) -> Tuple[int, int]:
    effective_batch_size = batch_size * accumulate_grad_batches
    steps_per_epoch = (train_samples_length // effective_batch_size) + 1
    total_training_steps = int(steps_per_epoch * n_epochs)
    warmup_steps = int(total_training_steps * warmup_ratio)
    return warmup_steps, total_training_steps


def construct_word_embedding(model, model_backbone: ModelBackboneTypes, input_ids: Tensor):
    backbone_name = BackbonesMetaData.name[model_backbone]
    model = getattr(model, backbone_name)
    if is_model_encoder_only(model_backbone):
        return model.embeddings.word_embeddings(input_ids)
    else:
        return model.get_input_embeddings()(input_ids)


def get_explained_ref_token_name(explained_tokenizer):
    if ExpArgs.ref_token_name == RefTokenNameTypes.MASK.value:
        return explained_tokenizer.mask_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.PAD.value:
        return explained_tokenizer.pad_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
        return explained_tokenizer.unk_token_id
    else:
        raise ValueError("ref name invalid")


def is_add_label_embedding():
    return ExpArgs.interpreter_label_token_position != LabelTokenPosition.NONE.value


def init_trainable_embeddings():
    task = ExpArgs.task
    interpreter_config = get_interpreter_config()
    label_embedding_index = None
    if is_add_label_embedding():
        num_label_embeddings = len(task.labels_str_int_maps.keys())
        if ExpArgs.is_include_general_label_token:
            num_label_embeddings = num_label_embeddings + 1
            label_embedding_index = torch.tensor(num_label_embeddings - 1)  # last item index
        trainable_embeddings = nn.Embedding(num_label_embeddings, interpreter_config.hidden_size,
                                            padding_idx = interpreter_config.pad_token_id)
        trainable_embeddings.weight.data.normal_(mean = 0.0, std = interpreter_config.initializer_range)

        for p in trainable_embeddings.parameters():
            p.requires_grad = True

        trainable_embeddings.train()

        return trainable_embeddings, label_embedding_index


def load_trainable_embeddings(trainable_embeddings):
    trainable_embeddings.eval()
    file_path = f"{ExpArgs.fine_tuned_interpreter_model_path}/{NEW_ADDED_TRAINABLE_PARAMS}.pth"
    if is_add_label_embedding():
        if Path(file_path).is_file():
            checkpoint = torch.load(file_path)
            trainable_embeddings.load_state_dict(checkpoint[NEW_ADDED_TRAINABLE_PARAMS])

            for p in trainable_embeddings.parameters():
                p.requires_grad = False
        else:
            raise ValueError("can not find saved embeddings")


def custom_teardown(trainer) -> None:
    self = trainer.strategy
    _optimizers_to_device(self.optimizers, torch.device("cpu"))

    if self.lightning_module is not None:
        self.lightning_module.interpreter_model.cpu()
    self.precision_plugin.teardown()
    self.accelerator.teardown()
    self.checkpoint_io.teardown()


def run_trainer(trainer, model, data_module, explained_model):
    gc.collect()
    torch.cuda.empty_cache()
    explained_model.zero_grad()


    trainer.fit(model = model, datamodule = data_module)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    explained_model.zero_grad()
