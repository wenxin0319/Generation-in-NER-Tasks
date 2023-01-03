import os
import json
import time
import random
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from model import StructuralModel
from config import Config
from data import IEDataset
from scorer import *
import copy
import ipdb
from util import (Logger, generate_vocabs)

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-w', '--weight_path')
args = parser.parse_args()
config = Config.from_json_file(args.config)
print(config.to_dict())
if type(config) is dict:
    config = Config.from_dict(config)

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
if config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

test_set = IEDataset(config.test_file, max_length=config.max_length, use_type=config.use_type)
# dev_set = IEDataset(os.path.join(config.dev_folder, 'dev.txt'), max_length=config.max_length, use_type=config.use_type)

eval_batch_size = config.eval_batch_size
map_location = f'cuda:{config.gpu_device}'
print(f"Loading model from {args.weight_path}")

state = torch.load(args.weight_path, map_location=map_location)

vocabs = state['vocabs']
# config = state['config']

# if type(config) is dict:
#     config = Config.from_dict(config)

model_name = config.bert_model_name
tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir)
test_set.numberize(tokenizer, vocabs)
# dev_set.numberize(tokenizer, vocabs)
output_space = len(vocabs['entity_label'])

test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)
# dev_batch_num = len(dev_set) // config.eval_batch_size + \
#     (len(dev_set) % config.eval_batch_size != 0)

model = StructuralModel(config, vocabs)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
model.load_state_dict(state['model'])
model.cuda(device=config.gpu_device)

all=[]
with open('6foldgroup_group4.json', 'w',encoding='utf-8') as f2:
    test_gold_entities, test_pred_entities, test_tokens = [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=test_set.collate_fn):
        pred_entities, _ = model.predict(batch)
        test_gold_entities.extend(batch.entity_lists)
        test_pred_entities.extend(pred_entities)
        test_tokens.extend(batch.tokens)

        assert len(pred_entities) == len(batch.tokens)

        for i in range(len(pred_entities)):
            _new={}
            _new["pred object"]=pred_entities[i]
            _new["tokens"]=batch.tokens[i]
            all.append(_new)

    json.dump(all,f2)
            
print("test")
precision, recall, f1, details = score_graphs(test_gold_entities, test_pred_entities)