# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py


from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments for the Trainer.
    """
    output_dir: str = field(
        default='experiments',
        metadata={"help": "The output directory where the results and model weights will be written."}
    )
    
    zero_shot: bool = field(
        default=False,
        metadata={"help": "Zero-shot setting"}
    )
    tf_log: str = field(
        default='./tf_log',
        metadata={"help": "The output directory for tensorboard."}
    ) 
    early_stopping_patience: int = field(
        default=7,
        metadata={"help": "The early stopping patience of the training process."}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load best model at end, to fit for earlystopping callback you should assign this as true."}
    )
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "The evaluation strategy for the training process"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names, for training."}
    )

    eval_dataset_during_train: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of evaluating dataseta names, for evaluating during training."}
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data directory"}
    )

    eval_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma separated list of dataset names. Defaults to the train datasets."}
    )

    train_split: str = field(
        default='train',
        metadata={"help": "The datasplit for training. Can be 'train', 'dev', 'test', etc."}
    )

    eval_split: str = field(
        default='dev',
        metadata={"help":"The datasplit for evaluating. Can be 'train', 'dev', 'test', etc."}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, shorter sequences will be padded."
        },
    )

    max_output_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length (default is the same as input)"
        },
    )

    static_class_choice: str = field(
        default='["person", "location", "organization", "geographical entity", "weapon", "vehicle"]',
        metadata={
            "help": "the class you want to train or test on (hyper definition of classes). In Continual Learning task, we assume you would add the class according to the seq you input."
        },
    )

    add_class_choice: str = field(
        default='["facility"]',
        metadata={
            "help": "the class you want to see after removing the fewshot class"
        },
    )

    mapping_input_choice: str = field(
        default='{"PER":"person", "LOC":"location", "ORG":"organization", "VEH":"vehicle", \
                "GPE":"geographical entity", "FAC":"facility", "WEA":"weapon"}',
        metadata={
            "help": "mapping initiation input choice"
        },
    )

    mapping_output_choice: str = field(
        default='{"PER":"person", "LOC":"location", "ORG":"organization", "VEH":"vehicle", \
                "GPE":"geographical entity", "FAC":"facility", "WEA":"weapon"}',
        metadata={
            "help": "mapping initiation output choice"
        },
    )

    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    train_subset: float = field(
        default=1, metadata={"help": "The portion of training data to use"}
    )

    eval_subset: float = field(
        default=1, metadata={"help": "The portion of evaluating data to use"}
    )

    episodes: str = field(
        default='0', metadata={"help": "Episode indices -- a single number such as 3 or an interval such as 1-4\n"
                                       "The index is also used as random seeds and this setting is therefore used to "
                                       "repeat multiple experiments."}
    )

    num_beams: int = field(
        default=None,
        metadata={"help": "Number of beams for beam search during generation (only affects evaluation)"}
    )

    max_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "Maximum input sequence length at evaluation time (default is equal to max_seq_length)"
        },
    )

    max_output_seq_length_eval: int = field(
        default=None,
        metadata={
            "help": "The maximum output sequence length at evaluation time (default is the same as input)"
        },
    )
    
    input_format: str = field(
        default=None, metadata={"help": "Input format"}
    )
    
    output_format: str = field(
        default=None, metadata={"help": "Output format"}
    )

    multitask: bool = field(
        default=False, metadata={"help": "If true, each input sentence is prepended with the dataset name"}
    )

    # few-shot arguments
    num_shots: int = field(
        default=None, metadata={"help": "number of shots (few-shot argument for the FewRel dataset)"}
    )

    num_ways: int = field(
        default=None, metadata={"help": "number of ways (few-shot argument for the FewRel dataset)"}
    )

    num_query: int = field(
        default=5, metadata={"help": "number of query examples (few-shot argument for the FewRel dataset)"}
    )

    # chunk arguments (used for the CoNLL2012 coreference resolution dataset)
    chunk_size: int = field(
        default=128, metadata={"help": "Size of document chunks"}
    )

    chunk_overlap: int = field(
        default=64, metadata={"help": "Size of overlap between consecutive chunks"}
    )

    chunk_size_eval: int = field(
        default=None, metadata={"help": "Size of document chunks during evaluation (default is equal to chunk_size)"}
    )

    chunk_overlap_eval: int = field(
        default=None, metadata={"help": "Size of overlap between consecutive chunks during evaluation "
                                        "(default is equal to chunk_overlap)"}
    )

    eval_nll: bool = field(
        default=False, metadata={"help": "Evaluate using NLL (only applicable to certain datasets)"}
    )
