"""Kurtis-E1: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from .config import DEFAULT_INSTRUCTION

FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    """Construct prompts."""
    output_texts = []
    for i in range(len(example["question"])):
        text = (
            f"<|im_start|>system\n{DEFAULT_INSTRUCTION.strip()}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{example['question'][i].strip()}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{example['answer'][i].strip()}"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
