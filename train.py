import logging
import sys
import os
import numpy as np
import jellyfish
from statistics import mode
import json
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartTokenizer,
    set_seed,
)
from typing import List, Optional
from datasets import load_from_disk,load_dataset, DatasetDict
from collections import defaultdict
from functools import partial
from dataclasses import dataclass, field
import transformers
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
logger = logging.getLogger(__name__)


def check_splits(ds):

    """

    Проверяет наличие train и test в DatasetDict.

    """

    if hasattr(ds, 'keys'):

        return 'train' in ds, 'test' in ds, 'validation' in ds,

    else:

        return False, False, False
        
def data_processin(example,is_training=False,tokenizer = None,padding = False,ignore_pad_token_for_loss=False):
    if tokenizer == None:
        return example
    
    if is_training:
        model_input = tokenizer(example['question'],truncation=False)
        if len(model_input['input_ids']) <= tokenizer.model_max_length:
            if padding == "max_length" and ignore_pad_token_for_loss:
                model_input['labels'] = [(l if l != tokenizer.pad_token_id else -100) 
                     for l in tokenizer(example['graph_answer'],
                        padding=padding, truncation=False)['input_ids']
                    ]
                return model_input
        else:
            model_input['input_ids'] = []
            model_input['labels'] = []
            return model_input 
            #print('Перебор')
    else:
        model_input = tokenizer(example['question'],padding=padding, truncation=False)
        if len(model_input['input_ids']) <= tokenizer.model_max_length:
            model_input['labels'] = [(l if l != tokenizer.pad_token_id else -100) 
                     for l in tokenizer(example['graph_answer'],
                        padding=padding, truncation=False)['input_ids']
                    ]
            return model_input
        else:
            model_input['input_ids'] = []
            model_input['labels'] = []
            return model_input


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name. "
                    "By default we use BART-large tokenizer for TAPEX-large."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    auth_token:  Optional[str] = field(
        default=None, metadata={"help": "Auth token for huggingface"}
    )
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    Omega_include: Optional[str] = field(
        default='pcsgbhobaopl', metadata={"help": "to get squall ids. Choose from ['p', 'pc', 'pcs', 'pcsgb', 'pcsgbh', 'pcsgbhob', 'pcsgbhoba', 'pcsgbhobaop', 'pcsgbhobaopl']"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    castom_data_local_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input local dataset dir"
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )

    def __post_init__(self):

        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.castom_data_local_dir is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = None
    logging.info('overwrite_output_dir set to False to avoid losing previous fine-tuning')
    overwrite_output_dir = False
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir :
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        try : 
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        except Exception as e:
            logger.info(f"Except load_dataset : {e} \ntry load from disk ..")
            datasets = load_from_disk(data_args.dataset_name)
            logger.info(f"Load dataset From Disk PATH = {data_args.dataset_name}")
    elif data_args.castom_data_local_dir is not None:
        datasets = load_from_disk(data_args.castom_data_local_dir)
        if not any(check_splits(datasets)):
            print("FAIL")
            datasets = DatasetDict({'train':datasets})
            print(datasets)
            #datasets = datasets.train_test_split(test_size=0.2, shuffle=True,seed=training_args.seed)
            #datasets['train'],datasets['validation'] = datasets['train'].train_test_split(test_size=0.25, 
            #                                                                           shuffle=True,
            #                                                                       seed=training_args.seed).values()
        
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    logger.info("datasets",datasets)
    logger.info("Load conf")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.auth_token if model_args.use_auth_token else None,
    )
    config.no_repeat_ngram_size = 0
    config.max_length = 1024
    config.early_stopping = False
    logger.info("Load tokenizer")

    tokenizer = BartTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.auth_token if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
    
    logger.info("Load model")

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.auth_token if model_args.use_auth_token else None,
    )
    padding = "max_length" if data_args.pad_to_max_length else False
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    data_processin_training = partial(data_processin, 
                                      is_training = training_args.do_train,
                                      tokenizer = tokenizer,padding = padding,
                                      ignore_pad_token_for_loss = data_args.ignore_pad_token_for_loss)
    data_processin_test = partial(data_processin, 
                                      is_training = False,
                                      tokenizer = tokenizer)


    
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            data_processin_training,
            #batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        train_dataset = train_dataset.filter(lambda x: True if len(x['input_ids']) > 0 else False,
                                             num_proc=data_args.preprocessing_num_workers)
        logger.info(f"Actual train size {len(train_dataset)}")
        #print("answers decode : ",tokenizer.decode(train_dataset["labels"][rnd]))

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            data_processin_training,
            #batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        eval_dataset = eval_dataset.filter(lambda x: True if len(x['input_ids']) > 0 else False,
                                             num_proc=data_args.preprocessing_num_workers)
        logger.info(f"Actual eval size {len(eval_dataset)}")
        
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            data_processin_test,
            #batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        predict_dataset = predict_dataset.filter(lambda x: True if len(x['input_ids']) > 0 else False,
                                             num_proc=data_args.preprocessing_num_workers)
        logger.info(f"Actual test size {len(predict_dataset)}")
        
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        delimiter = ", "
        
        # define example evaluation
        def evaluate_example(predict_str: str, ground_str: str):
            predict_spans = predict_str.split(delimiter)
            ground_spans = ground_str.split(delimiter)
            predict_values = defaultdict(lambda: 0)
            ground_values = defaultdict(lambda: 0)
            for span in predict_spans:
                try:
                    predict_values[float(span)] += 1
                except ValueError:
                    predict_values[span.strip()] += 1
            for span in ground_spans:
                try:
                    ground_values[float(span)] += 1
                except ValueError:
                    ground_values[span.strip()] += 1
            _is_correct = predict_values == ground_values
            return _is_correct

        def get_denotation_accuracy_and_levenshtein_dist(predictions: List[str], references: List[str]):
            assert len(predictions) == len(references)
            correct_num = 0
            lev_dist = []
            for predict_str, ground_str in zip(predictions, references):
                lev_dist.append(jellyfish.levenshtein_distance(predict_str.lower().strip(), ground_str.lower().strip()))
                is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
                if is_correct:
                    correct_num += 1
            output_prediction_file = os.path.join(training_args.output_dir, "tapexgraph_outputs.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join([f"pred: {x} ref: {y}" for x,y in zip(predictions, references)]))
            return correct_num / len(predictions), np.mean(lev_dist), np.std(lev_dist),mode(lev_dist)
        
        accuracy, mean, std, l_mode = get_denotation_accuracy_and_levenshtein_dist(decoded_preds, decoded_labels)
        result = {"denotation_accuracy": accuracy,
                 "levenshtein_dist_mean": mean,
                 "levenshtein_dist_std":std,
                "levenshtein_dist_mode":l_mode
                 }
        
        #with open(model_args.model_name_or_path/
        return result

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    #results = {}
    #if training_args.do_eval:
    #    logger.info("*** Last Evaluate ***")

#        metrics = trainer.evaluate(
 #           max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="last_eval"
  #      )
   #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
#
 #       trainer.log_metrics("last_eval", metrics)
  #      trainer.save_metrics("last_eval", metrics)
#
    if training_args.do_predict:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            )
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "tapex_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    return results

if __name__ == "__main__":
    main()
