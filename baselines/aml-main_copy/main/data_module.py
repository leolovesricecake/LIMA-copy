import pytorch_lightning as pl
import tokenizations
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

from config.config import ExpArgs
from config.constants import (LABELS_NAME, EXPLAINED_INPUT_IDS_NAME, EXPLAINED_ATTENTION_MASK_NAME,
                              INTERPRETER_ATTENTION_MASK_NAME, INTERPRETER_INPUT_IDS_NAME, MAP_TOKENS, TEXT_PROMPT,
                              TASK_PROMPT_INPUT_IDS, LABEL_PROMPT_INPUT_IDS, LABEL_PROMPT_ATTENTION_MASK,
                              TASK_PROMPT_ATTENTION_MASK, LABEL_PROMPT_NEW_LINE, INPUT_TXT)
from config.types_enums import ValidationType
from models.train_models_utils import get_models_tokenizer
from utils.utils_functions import is_model_encoder_only, is_use_prompt, get_model_special_tokens


class DataModule(pl.LightningDataModule):
    def __init__(self, val_type: ValidationType, train_sample: int = -1, test_sample: int = -1,
                 explained_tokenizer = None, interpreter_tokenizer = None, data = None, task_prompt_input_ids = None,
                 task_prompt_attention_mask = None, label_prompt_input_ids = None, label_prompt_attention_mask = None):
        super().__init__()
        self.task = ExpArgs.task
        self.seed = ExpArgs.seed
        self.train_dataset, self.val_dataset = None, None
        self.val_type = val_type.value
        self.train_sample = train_sample
        self.test_sample = test_sample
        self.task_prompt = None
        self.input_prompt = None
        self.pre_label_prompt = None
        self.task_prompt_input_ids = task_prompt_input_ids
        self.task_prompt_attention_mask = task_prompt_attention_mask
        self.label_prompt_input_ids = label_prompt_input_ids
        self.label_prompt_attention_mask = label_prompt_attention_mask
        if explained_tokenizer is not None:
            self.interpreter_tokenizer = interpreter_tokenizer
            self.explained_tokenizer = explained_tokenizer
        else:
            self.interpreter_tokenizer = get_models_tokenizer(ExpArgs.interpreter_model_backbone)
            self.explained_tokenizer = get_models_tokenizer(ExpArgs.explained_model_backbone)
            self.set_label_vocab_tokens()

        self.interpreter_tokenizer_special_token_ids = torch.tensor(
            get_model_special_tokens(ExpArgs.interpreter_model_backbone, self.interpreter_tokenizer))

        # SET MAX LENGTH
        if self.task.is_llm_set_max_len and (not is_model_encoder_only(ExpArgs.explained_model_backbone)):
            self.explained_tokenizer.model_max_length = self.task.llm_explained_tokenizer_max_length
            self.interpreter_tokenizer.model_max_length = self.task.llm_interpreter_tokenizer_max_length

        if data:
            for k, v in data.items():
                if k != INPUT_TXT:
                    data[k] = data[k].unsqueeze(0)
                elif k == INPUT_TXT:
                    data[k] = [data[k]]
            self.train_dataset = Dataset.from_dict(data)
            self.val_dataset = Dataset.from_dict(data)
            self.train_dataset.set_format(type = 'torch', columns = list(self.train_dataset.features))
            self.val_dataset.set_format(type = 'torch', columns = list(self.val_dataset.features))
        else:
            self.dataset = load_dataset(self.task.dataset_name)
            self.dataset_column_text = self.task.dataset_column_text
            self.dataset_column_label = self.task.dataset_column_label
            self.setup()

    def setup(self, stage = None):

        if not self.train_dataset:
            if is_use_prompt():
                self.set_prompt()

            self.setup_train_ds()
            self.setup_test_ds()

    def setup_train_ds(self):
        tmp_train_ds = self.dataset[self.task.dataset_train].shuffle(seed = self.seed)
        if self.train_sample:
            tmp_train_ds = tmp_train_ds.train_test_split(train_size = self.train_sample, seed = self.seed,
                                                         stratify_by_column = self.dataset_column_label)
            tmp_train_ds = tmp_train_ds["train"]
        self.train_dataset = self.handle_ds(tmp_train_ds)

    def setup_test_ds(self):
        if self.val_type == ValidationType.VAL.value:
            tmp_test_ds = self.dataset[self.task.dataset_val].shuffle(seed = self.seed)
            if self.test_sample:
                tmp_test_ds = tmp_test_ds.train_test_split(test_size = self.test_sample, seed = self.seed,
                                                           stratify_by_column = self.dataset_column_label)
                tmp_test_ds = tmp_test_ds["test"]
        else:
            tmp_test_ds = self.dataset[self.task.dataset_test].shuffle(seed = self.seed)
            if self.test_sample:
                tmp_test_ds = tmp_test_ds.train_test_split(train_size = self.test_sample, seed = self.seed,
                                                           stratify_by_column = self.dataset_column_label)
                tmp_test_ds = tmp_test_ds["train"]

        self.val_dataset = self.handle_ds(tmp_test_ds)

    def handle_ds(self, ds):
        ds = ds.map(self.tokenize, batched = False)
        ds = ds.remove_columns(self.dataset_column_text)
        ds = ds.rename_column(self.dataset_column_label, LABELS_NAME)
        if "id" not in ds.features.keys():
            ds = ds.add_column("id", [i for i in range(ds.num_rows)])
        ds.set_format(type = 'torch', columns = ds.features)
        return ds

    def set_label_vocab_tokens(self):
        if is_use_prompt():
            labels_tokens = [self.explained_tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False)
                             for l in list(ExpArgs.task.labels_int_str_maps.keys())]
            ExpArgs.label_vocab_tokens = torch.stack(labels_tokens).squeeze()
            if ExpArgs.label_vocab_tokens.ndim != 1:
                raise ValueError("label_vocab_tokens must work with one token only")

    def tokenize(self, example):
        inputs_txt = example[self.dataset_column_text]
        if is_use_prompt():
            inputs_txt = self.input_prompt + inputs_txt
            explained_tokenized_input = self.explained_tokenizer.encode_plus(inputs_txt, truncation = True,
                                                                             add_special_tokens = False)
        else:
            explained_tokenized_input = self.explained_tokenizer.encode_plus(inputs_txt, truncation = True,
                                                                             add_special_tokens = True)

        explained_tokenized_input_dict = {EXPLAINED_INPUT_IDS_NAME: explained_tokenized_input.input_ids,
                                          EXPLAINED_ATTENTION_MASK_NAME: explained_tokenized_input.attention_mask}

        # interpreter
        interpreter_tokenized_input = self.interpreter_tokenizer.encode_plus(inputs_txt, truncation = True,
                                                                             add_special_tokens = True)
        interpreter_tokenized_input_dict = {INTERPRETER_INPUT_IDS_NAME: interpreter_tokenized_input.input_ids,
                                            INTERPRETER_ATTENTION_MASK_NAME: interpreter_tokenized_input.attention_mask}

        tokenized_input = {**explained_tokenized_input_dict, **interpreter_tokenized_input_dict}
        tokenized_input[INPUT_TXT] = inputs_txt
        return tokenized_input

    def train_dataloader(self):
        return DataLoader(dataset = self.train_dataset, batch_size = ExpArgs.batch_size, shuffle = True,
                          collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = ExpArgs.eval_batch_size, shuffle = False,
                          collate_fn = self.collate_fn)

    @staticmethod
    def convert_to_token(item_val, tokenizer):
        val = item_val
        if isinstance(item_val, torch.Tensor):
            val = item_val.item()
        return tokenizer.convert_ids_to_tokens(val)

    def set_prompt(self):
        if is_use_prompt():

            task_prompt = self.task.llm_task_prompt
            few_shots_prompt = self.task.llm_few_shots_prompt

            self.task_prompt = "\n\n".join([task_prompt, few_shots_prompt, TEXT_PROMPT])
            self.input_prompt = ""

            tmp_tokenized = self.explained_tokenizer.encode_plus(self.task_prompt, return_tensors = "pt",
                                                                 add_special_tokens = True)

            self.task_prompt_input_ids = tmp_tokenized.input_ids
            self.task_prompt_attention_mask = tmp_tokenized.attention_mask

            tmp_tokenized = self.explained_tokenizer.encode_plus(LABEL_PROMPT_NEW_LINE, return_tensors = "pt",
                                                                 add_special_tokens = False)

            self.label_prompt_input_ids = tmp_tokenized.input_ids.squeeze()
            self.label_prompt_attention_mask = tmp_tokenized.attention_mask.squeeze()

    @staticmethod
    def pad_sequences(key, batch, tokenizer, model_backbone, is_inputs_ids, maps = None, is_return_maps = False):
        sequences = [item[key] for item in batch]

        if not is_model_encoder_only(model_backbone) and is_use_prompt():
            if is_return_maps:
                return sequences, maps
            return sequences

        max_len = max([seq.shape[-1] for seq in sequences])
        pad = tokenizer.pad_token_id if is_inputs_ids else 0  # 0 for attention_mask

        if is_model_encoder_only(model_backbone):
            padded_sequences = [torch.cat(  #
                (seq, torch.tensor([pad] * (max_len - len(seq)))),  #
                dim = 0).int() for seq in
                                sequences]  # padded_sequences = [seq + [pad] * (max_len - len(seq)) for seq in sequences]
        else:
            padded_sequences = [  #
                torch.cat(  #
                    (torch.tensor([pad] * (max_len - len(seq))), seq),  #
                    dim = 0).int() for seq in sequences]
            if is_return_maps:
                maps = [[[-1]] * (max_len - len(seq)) + maps[seq_idx] for seq_idx, seq in enumerate(sequences)]
                if maps is None:
                    raise ValueError("maps is None")
                return torch.stack(padded_sequences).long(), maps

        if is_return_maps:
            return torch.stack(padded_sequences).long(), maps
        return torch.stack(padded_sequences).long()

    def pad_task_prompts_sequences(self, batch, tokenizer):
        if not is_use_prompt():
            return None, None
        sequences = [self.task_prompt_input_ids.squeeze().tolist() + item[EXPLAINED_INPUT_IDS_NAME].tolist() for item in
                     batch]
        max_len = max([len(seq) for seq in sequences])

        padded_task_prompts_input_ids = [torch.cat(  #
            (torch.tensor([tokenizer.pad_token_id] * (max_len - len(seq))),  #
             self.task_prompt_input_ids.squeeze()),  #
            dim = 0).long() for seq in sequences]

        padded_task_prompts_attention_mask = [torch.cat(  #
            (torch.tensor([0] * (max_len - len(seq))),  #
             self.task_prompt_attention_mask.squeeze()),  #
            dim = 0).long() for seq in sequences]

        return padded_task_prompts_input_ids, padded_task_prompts_attention_mask

    def collate_fn(self, batch):
        input_texts = [item[INPUT_TXT] for item in batch]
        # map tokens
        maps = self.build_tokenizer_relations(batch)

        explained_input_ids, maps = self.pad_sequences(EXPLAINED_INPUT_IDS_NAME, batch, self.explained_tokenizer,
                                                       ExpArgs.explained_model_backbone, is_inputs_ids = True,
                                                       maps = maps, is_return_maps = True)

        interpreter_input_ids = self.pad_sequences(INTERPRETER_INPUT_IDS_NAME, batch, self.interpreter_tokenizer,
                                                   ExpArgs.interpreter_model_backbone, is_inputs_ids = True)
        padded_task_prompts_input_ids, padded_task_prompts_attention_mask = self.pad_task_prompts_sequences(batch,
                                                                                                            self.explained_tokenizer)
        return {EXPLAINED_INPUT_IDS_NAME: explained_input_ids,
                EXPLAINED_ATTENTION_MASK_NAME: self.pad_sequences(EXPLAINED_ATTENTION_MASK_NAME, batch,
                                                                  self.explained_tokenizer,
                                                                  ExpArgs.explained_model_backbone,
                                                                  is_inputs_ids = False),
                INTERPRETER_INPUT_IDS_NAME: interpreter_input_ids,
                INTERPRETER_ATTENTION_MASK_NAME: self.pad_sequences(INTERPRETER_ATTENTION_MASK_NAME, batch,
                                                                    self.interpreter_tokenizer,
                                                                    ExpArgs.interpreter_model_backbone,
                                                                    is_inputs_ids = False),  #
                MAP_TOKENS: maps,  #
                TASK_PROMPT_INPUT_IDS: padded_task_prompts_input_ids,  #
                LABEL_PROMPT_INPUT_IDS: self.label_prompt_input_ids,  #
                TASK_PROMPT_ATTENTION_MASK: padded_task_prompts_attention_mask,  #
                LABEL_PROMPT_ATTENTION_MASK: self.label_prompt_attention_mask,  #
                INPUT_TXT: input_texts  #
                }

    def fill_empty_items(self, lists):
        for i in range(len(lists)):
            if not lists[i]:
                prev_idx = next((j for j in range(i - 1, -1, -1) if lists[j]), None)
                next_idx = next((j for j in range(i + 1, len(lists)) if lists[j]), None)

                prev_max = max(lists[prev_idx]) if prev_idx is not None else None
                next_min = min(lists[next_idx]) if next_idx is not None else None

                if prev_max is not None and next_min is not None:
                    diff = next_min - prev_max
                    if diff == 1:
                        lists[i] = [prev_max + 1]
                    elif diff == 0:
                        lists[i] = [prev_max]
                    elif diff > 1 and next_idx - prev_idx == 2:
                        lists[i] = list(range(prev_max + 1, next_min))
                    else:
                        gap = (next_min - prev_max - 1) // (next_idx - prev_idx - 1)
                        lists[i] = [prev_max + gap * (i - prev_idx)]
                elif prev_max is not None:
                    lists[i] = [prev_max + 1]
                elif next_min is not None:
                    lists[i] = [next_min - 1]

        return lists

    def build_tokenizer_relations(self, batch):
        maps = []
        if is_model_encoder_only(ExpArgs.explained_model_backbone):
            return None
        for item in batch:

            explained_tokens = [self.convert_to_token(i, self.explained_tokenizer) for i in
                                item[EXPLAINED_INPUT_IDS_NAME]]
            interpreter_tokens = [self.convert_to_token(i, self.interpreter_tokenizer) for i in
                                  item[INTERPRETER_INPUT_IDS_NAME]]
            a2b, b2a = tokenizations.get_alignments(explained_tokens, interpreter_tokens)

            a2b = self.fill_empty_items(a2b)

            if not explained_tokens[-1]:
                explained_tokens[-1] = len(interpreter_tokens) - 2

            special_token_indices = torch.nonzero(
                torch.isin(item[INTERPRETER_INPUT_IDS_NAME], self.interpreter_tokenizer_special_token_ids),
                as_tuple = False).squeeze().tolist()

            a2b = [[num for num in sublist if num not in special_token_indices] for sublist in a2b]
            maps.append(a2b)

        return maps
