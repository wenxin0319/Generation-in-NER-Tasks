import os
import json
import time
import random
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, AdamW,
                          RobertaTokenizer,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup)
from model import StructuralModel
from config import Config
from data import IEDataset
from scorer import *
import copy
import ipdb
from util import (Logger, generate_vocabs)


# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True )
parser.add_argument('-w', '--weight_path', required=True)
args = parser.parse_args()
config = Config.from_json_file(args.config)
print(config.to_dict())

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
if config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# load weight
map_location = f'cuda:{config.gpu_device}'
print(f"Loading model from {args.weight_path}")
state = torch.load(args.weight_path, map_location=map_location)
gpu_device = config.gpu_device

# load dataset
print('==============Prepare Dev Set=================')
dev_set = IEDataset(config.dev_file, max_length=config.max_length, use_type=config.use_type)
print('==============Prepare Test Set=================')
test_set = IEDataset(config.test_file, max_length=config.max_length, use_type=config.test_use_type)
vocabs = state['vocabs']

eval_batch_size = config.eval_batch_size
config = state['config']
if type(config) is dict:
    config = Config.from_dict(config)

model_name = config.bert_model_name
if model_name.startswith('bert-'):
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)
elif model_name.startswith('roberta-'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False,
                                          use_fast=False)

print('==============Prepare Dev Set=================')
dev_set.numberize(tokenizer, vocabs)
print('==============Prepare Test Set=================')
test_set.numberize(tokenizer, vocabs)

dev_batch_num = len(dev_set) // eval_batch_size + \
    (len(dev_set) % eval_batch_size != 0)
test_batch_num = len(test_set) // eval_batch_size + \
    (len(test_set) % eval_batch_size != 0)

# initialize the model
model = StructuralModel(config, vocabs)
model.load_state_dict(state['model'])
model.cuda(device=gpu_device)
model.eval()

# Evaluation 
# dev set
progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                    desc='Dev')
dev_gold_entities, dev_pred_entities, dev_tokens = [], [], []
for batch in DataLoader(dev_set, batch_size=eval_batch_size,
                        shuffle=False, collate_fn=dev_set.collate_fn):
    progress.update(1)
    pred_entities, _ = model.predict(batch)
    dev_gold_entities.extend(batch.entity_lists)
    dev_pred_entities.extend(pred_entities)
    dev_tokens.extend(batch.tokens)
#ipdb.set_trace()
progress.close()
precision, recall, f1, details = score_graphs(dev_gold_entities, dev_pred_entities)

# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                    desc='Test')
test_gold_entities, test_pred_entities, test_tokens = [], [], []
for batch in DataLoader(test_set, batch_size=eval_batch_size,
                        shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    pred_entities, _ = model.predict(batch)
    test_gold_entities.extend(batch.entity_lists)
    test_pred_entities.extend(pred_entities)
    test_tokens.extend(batch.tokens)
progress.close()
precision, recall, f1, details = score_graphs(test_gold_entities, test_pred_entities)
