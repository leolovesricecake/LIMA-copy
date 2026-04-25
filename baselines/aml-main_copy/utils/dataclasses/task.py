from typing import Union, Dict, List, Tuple

from config.constants import TEXT_PROMPT, LABEL_PROMPT


class Task:
    def __init__(self, dataset_name: str, dataset_train: str, dataset_val: str, dataset_test: str,
                 dataset_column_text: str, dataset_column_label: str, bert_fine_tuned_model: str,
                 roberta_fine_tuned_model: str, distilbert_fine_tuned_model: str, roberta_base_model: str,
                 distilbert_base_model: str, bert_base_model: str, llama_model: str, mistral_model: str,
                 labels_str_int_maps: Union[Dict, None], default_lr: float, llm_lr: float,
                 test_sample: Union[int, None], train_sample: Union[int, None], hp_search_test_sample: Union[int, None],
                 hp_search_train_sample: Union[int, None], name: str, paper_name: str, is_finetuned_with_lora: bool,
                 hp_search_n_trials: int, llm_task_prompt: str, llm_few_shots_prompt: List[Tuple[str, str]],
                 is_llm_set_max_len = False, llama_adapter: str = None, mistral_adapter: str = None,
                 llm_explained_tokenizer_max_length: int = 0, llm_interpreter_tokenizer_max_length: int = -1):
        self.dataset_name = dataset_name
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.dataset_column_text = dataset_column_text
        self.dataset_column_label = dataset_column_label
        self.bert_fine_tuned_model = bert_fine_tuned_model
        self.roberta_fine_tuned_model = roberta_fine_tuned_model
        self.distilbert_fine_tuned_model = distilbert_fine_tuned_model

        self.roberta_base_model = roberta_base_model
        self.distilbert_base_model = distilbert_base_model
        self.bert_base_model = bert_base_model

        self.llama_model = llama_model
        self.mistral_model = mistral_model
        self.llama_adapter = llama_adapter
        self.mistral_adapter = mistral_adapter

        self.is_llm_use_lora = is_finetuned_with_lora
        self.labels_str_int_maps = labels_str_int_maps
        self.labels_int_str_maps = {value: key for key, value in
                                    labels_str_int_maps.items()} if labels_str_int_maps else None
        self.default_lr = default_lr
        self.llm_lr = llm_lr
        self.test_sample = test_sample
        self.train_sample = train_sample
        self.val_sample = hp_search_test_sample
        self.hp_search_train_sample = hp_search_train_sample
        self.hp_search_n_trials = hp_search_n_trials
        self.name = name
        self.paper_name = paper_name
        self.is_llm_set_max_len = is_llm_set_max_len
        self.llm_explained_tokenizer_max_length = llm_explained_tokenizer_max_length
        self.llm_interpreter_tokenizer_max_length = llm_interpreter_tokenizer_max_length
        self.llm_task_prompt = llm_task_prompt
        # self.llm_few_shots_prompt = llm_few_shots_prompt  # for test only
        self.llm_few_shots_prompt = "\n\n".join(
            ["\n".join([TEXT_PROMPT + i[0], LABEL_PROMPT + str(i[1])]) for i in llm_few_shots_prompt])
