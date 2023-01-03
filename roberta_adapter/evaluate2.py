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

from model import StructuralModel,Adapter
from config import Config
from data import IEDataset
from scorer import *
import copy
import ipdb
from util import (Logger, generate_vocabs)

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-w', '--modeldir')
args = parser.parse_args()
config = Config.from_json_file(args.config)
# print(config.to_dict())
if type(config) is dict:
    config = Config.from_dict(config)

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

eval_batch_size = config.eval_batch_size

# set GPU device
if config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

model_path = os.path.join(args.modeldir, 'best.base.mdl')
adapter_path = os.path.join(args.modeldir, 'best.adapter.mdl')

model_name = config.bert_model_name
tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir)

test_set = IEDataset(config.test_file, max_length=config.max_length, use_type=config.use_type2)

test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

map_location = f'cuda:{config.gpu_device}'
state_model = torch.load(model_path, map_location=map_location)
state_adapter = torch.load(adapter_path, map_location=map_location)

vocabs0=state_model['vocabs']
vocabs1=state_adapter['vocabs']

test_set.numberize(tokenizer, vocabs1)
output_space = len(vocabs0['entity_label'])

model = StructuralModel(config,vocabs0)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
model.load_state_dict(state_model['model'])
model.cuda(device=config.gpu_device)
model.eval()

adapter = Adapter(config,vocabs1,output_space)
adapter.load_bert(model_name, cache_dir=config.bert_cache_dir)
adapter.load_state_dict(state_adapter['model'])
adapter.cuda(device=config.gpu_device)
adapter.eval()

allowed_type=config.use_type2
print(allowed_type)

all=[]
with open('955person.json', 'w',encoding='utf-8') as f2:
    test_gold_entities, test_pred_entities, test_tokens = [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=test_set.collate_fn):
        _, prior_logit = model.predict(batch)
        pred_entities, _ = adapter.predict(batch, prior_logit)
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