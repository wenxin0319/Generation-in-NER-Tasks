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
                          XLMRobertaTokenizer,
                          BartTokenizer,
                          MBart50Tokenizer,
                          MT5Tokenizer,
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
parser.add_argument('-c', '--config', default='config/example.json')
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

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_dir = os.path.join(config.log_path, timestamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = Logger(log_dir)
output_dir = os.path.join(config.output_path, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_role_model = os.path.join(output_dir, 'best.role.mdl')

model_name = config.bert_model_name
if model_name.startswith('bert-'):
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)
elif model_name.startswith('roberta-'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir)
elif model_name.startswith('xlm-roberta-'):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir)
elif model_name.startswith('lanwuwei'):
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                    cache_dir=config.bert_cache_dir, 
                                    do_lower_case=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False,
                                          use_fast=False)

# datasets
print('==============Prepare Training Set=================')
train_set = IEDataset(os.path.join(config.train_folder, 'train.txt'), max_length=config.max_length, use_type=config.use_type)
print('==============Prepare Dev Set=================')
dev_set = IEDataset(os.path.join(config.dev_folder, 'dev.txt'), max_length=config.max_length, use_type=config.use_type)
print('==============Prepare Test Set=================')
test_set = IEDataset(config.test_file, max_length=config.max_length, use_type=config.use_type)
vocabs = generate_vocabs([train_set, dev_set, test_set])
print(vocabs)

print('==============Prepare Training Set=================')
train_set.numberize(tokenizer, vocabs)
print('==============Prepare Dev Set=================')
dev_set.numberize(tokenizer, vocabs)
print('==============Prepare Test Set=================')
test_set.numberize(tokenizer, vocabs)

batch_num = len(train_set) // config.batch_size + \
    (len(train_set) % config.batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = StructuralModel(config, vocabs)
model.load_bert(model_name, cache_dir=config.bert_cache_dir)
model.cuda(device=config.gpu_device)

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    }
]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*config.warmup_epoch,
                                           num_training_steps=batch_num*config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs)

best_dev = 0.0
current_step = 0
best_detail = None
print('================Start Training================')
for epoch in range(config.max_epoch):
    print('Epoch: {}'.format(epoch))
    
    # training step
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):
        loss = model(batch, logger, 'train', current_step)
        current_step += 1
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()

    # dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                         desc='Dev {}'.format(epoch))
    best_dev_role_model = False
    dev_gold_entities, dev_pred_entities, dev_tokens = [], [], []
    for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        pred_entities, _ = model.predict(batch)
        dev_gold_entities.extend(batch.entity_lists)
        dev_pred_entities.extend(pred_entities)
        dev_tokens.extend(batch.tokens)
    #ipdb.set_trace()
    progress.close()
    precision, recall, f1, details = score_graphs(dev_gold_entities, dev_pred_entities)

    if f1 > best_dev:
        best_dev = f1
        print('Saving best model')
        torch.save(state, best_role_model)
        best_dev_role_model = True
    
    result = json.dumps(
        {'epoch': epoch, 'dev': (precision, recall, f1)}
    )
    with open(log_file, 'a', encoding='utf-8') as w:
        w.write(result + '\n')
    # tensorboard
    logger.scalar_summary('dev/entity', f1, epoch)

    if best_dev_role_model:    
        # test set
        progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                            desc='Test {}'.format(epoch))
        test_gold_entities, test_pred_entities, test_tokens = [], [], []
        for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=test_set.collate_fn):
            progress.update(1)
            pred_entities, _ = model.predict(batch)
            test_gold_entities.extend(batch.entity_lists)
            test_pred_entities.extend(pred_entities)
            test_tokens.extend(batch.tokens)
        progress.close()
        precision, recall, f1, details = score_graphs(test_gold_entities, test_pred_entities)
        best_detail = details

print(best_detail)
