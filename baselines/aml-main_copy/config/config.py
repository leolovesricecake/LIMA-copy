from config.types_enums import *
from config.types_enums import SchedulerTypes
from utils.dataclasses import Task


class ExpArgs:
    seed = 42
    num_epochs_for_pre_train = 10
    num_epochs_for_fine_tune = 10
    warmup_ratio = 0
    enable_checkpointing = False
    default_root_dir = "OUT"
    tokens_attr_with_ref_token_function_type: TokenTransformationTypes = TokenTransformationTypes.BLEND.value
    regularization_type: RegularizationTypes = RegularizationTypes.BCE.value
    ref_token_name: RefTokenNameTypes = RefTokenNameTypes.MASK.value
    lr = None
    start_epoch_to_evaluate = 0
    batch_size = 20
    accumulate_grad_batches = 1
    eval_batch_size = 1
    val_check_interval = 0.32
    is_save_model = False
    log_every_n_steps = 40
    inverse_token_attr_function = InverseLossTypes.NEGATIVE_PROB_LOSS.value
    eval_metric = None
    is_save_results = False
    is_save_support_results = False
    task: Task = None
    explained_model_backbone = None
    interpreter_model_backbone = None
    fine_tuned_interpreter_model_path = None
    run_type = None
    label_vocab_tokens = None
    scheduler_type = SchedulerTypes.LINEAR_SCHEDULE_WITH_WARMUP.value
    fine_tune_scheduler_type = SchedulerTypes.LINEAR_SCHEDULE_WITH_WARMUP.value
    interpreter_model_n_first_layers_to_freeze = None
    is_interpreter_model_train_mode_in_fine_tune = False
    interpreter_label_token_position = LabelTokenPosition.AFTER_LAST_SEP.value
    interpreter_classifier_size: int = 2
    interpreter_classifier_activation_function: ActivationFunctionTypes = ActivationFunctionTypes.TANH.value
    is_include_general_label_token = True
    llm_prompt_type: ModelPromptType = ModelPromptType.FEW_SHOT.value
    cross_tokenizers_pooling = CrossTokenizersPooling.MAX.value
    eval_tokens = EvalTokens.NO_SPECIAL_TOKENS.value
    is_inference = False
    hp_search_max_epochs = 2


class LossCoefficients:
    regularization_loss_weight = None
    prediction_loss_weight = None
    inverse_loss_weight = None


class MetricsMetaData:
    directions = {EvalMetric.SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.EVAL_LOG_ODDS.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_SUFFICIENCY.value: DirectionTypes.MIN.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS.value: DirectionTypes.MAX.value,
                  EvalMetric.AOPC_COMPREHENSIVENESS_AOPC_SUFFICIENCY.value: DirectionTypes.MAX.value,
                  EvalMetric.COMPREHENSIVENESS_SUFFICIENCY.value: DirectionTypes.MAX.value}

    top_k = {EvalMetric.SUFFICIENCY.value: [20], EvalMetric.COMPREHENSIVENESS.value: [20],
             EvalMetric.EVAL_LOG_ODDS.value: [20], EvalMetric.AOPC_SUFFICIENCY.value: [1, 5, 10, 20, 50],
             EvalMetric.AOPC_COMPREHENSIVENESS.value: [1, 5, 10, 20, 50]}


class BackbonesMetaData:
    name = {  #
        ModelBackboneTypes.BERT.value: "bert",  #
        ModelBackboneTypes.ROBERTA.value: "roberta",  #
        ModelBackboneTypes.DISTILBERT.value: "distilbert",  #
        ModelBackboneTypes.LLAMA.value: "model",  #
        ModelBackboneTypes.MISTRAL.value: "model"  #
    }
