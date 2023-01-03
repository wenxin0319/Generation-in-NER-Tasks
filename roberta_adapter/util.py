import os
import json
import glob
import lxml.etree as et
import numpy as np
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
from tensorboardX import SummaryWriter
import torch
import ipdb

class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)

    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    for t in sorted(entity_type_set):
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(sorted(entity_type_set), 1)}
    entity_type_stoi['O'] = 0
    entity_type_itos = {i:k for k, i in entity_type_stoi.items()}

    return {
        'entity_type': entity_type_stoi,
        'entity_label': entity_label_stoi,
        'entity_type_itos': entity_type_itos
    }

def update_vocabs(vocabs, datasets):
    ori_entities = vocabs['entity_type'].keys()
    
    entity_type_set = []
    for dataset in datasets:
        for e in dataset.entity_type_set:
            if e not in ori_entities:
                entity_type_set.append(e)
    entity_type_set = set(entity_type_set)
    prefix = ['B', 'I']
    entity_label_stoi = deepcopy(vocabs['entity_label'])
    for t in sorted(entity_type_set):
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)

    entity_type_stoi = deepcopy(vocabs['entity_type'])
    for t in sorted(entity_type_set):
        entity_type_stoi[t] = len(entity_type_stoi)

    entity_type_itos = {i:k for k, i in entity_type_stoi.items()}
    return {
        'entity_type': entity_type_stoi,
        'entity_label': entity_label_stoi,
        'entity_type_itos': entity_type_itos
    }