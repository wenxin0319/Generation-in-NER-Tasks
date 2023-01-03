import random
import ipdb
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)

DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

tokenizer = AutoTokenizer.from_pretrained(
    "t5-small"
)
# ADD special_tokens
special_tokens = ["[SEP]", "[CLS-SEP]"]
tokenizer.add_tokens(special_tokens)
def select_with_probability(features):
    feature_ret = []
    for feature in features:
        if random.random() < 0.8:
            feature_ret.append(feature)
    return feature_ret


def cut_with_probability(batch):
    labels_ret = []
    input_id_ret = []
    attention_ret = []
    dtype_label = torch.long # if type((batch["labels"].numpy())[0]) is int else torch.float
    dtype_inputid = torch.long # if type((batch["input_ids"].numpy())[0]) is int else torch.float
    dtype_attention = torch.long # if type((batch["attention_mask"].numpy())[0]) is int else torch.float

    batch_ret = {}
    ipdb.set_trace()
    bak_label_batch = []
    bak_inputid_batch = []
    bak_attention_batch = []
    for index in range( len(list(batch["labels"])) ):
        bak_label_batch.append("[SEP]")
        for label_id in range(0, os.environ["NUM_LABELS"]):
            if random.random() > 0.2:
                bak_label_batch.append(tokenizer.convert_ids_to_tokens(
                    batch["labels"][index].numpy()[2 * label_id + 1] ) )
                bak_label_batch.append("[CLS-SEP]")
            bak_label_batch.append(batch["input_ids"][index].numpy()[16:])
            labels_ret.append(batch["labels"][index].numpy())
            attention_ret.append(batch["attention_mask"][index].numpy())
    print(labels_ret)
    batch_ret["labels"] = torch.tensor(labels_ret, dtype=dtype_label)
    batch_ret["input_ids"] = torch.tensor(input_id_ret, dtype=dtype_inputid)
    batch_ret["attention_mask"] = torch.tensor(attention_ret, dtype=dtype_attention)
    return batch_ret

def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    # print("calling default data collator...")
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    # batch = cut_with_probability(batch)
    return batch


