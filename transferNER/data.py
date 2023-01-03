import copy
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
from itertools import combinations 
import ipdb

instance_fields = [
    'tokens', 'pieces', 'piece_idxs', 
    'token_lens', 'attention_mask',
    'entity_label_idxs', 'entity_list',
    'word_idxs', 'char_idxs'
]

batch_fields = [
    'tokens', 'piece_idxs', 
    'token_lens', 'attention_masks',
    'entity_label_idxs', 'entity_lists',
    'token_nums', 'word_idxs', 'char_idxs', 'char_nums'
]

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

def get_entity_labels(entities, token_num):
    """Convert entity mentions in a sentence to an entity label sequence with
    the length of token_num
    :param entities (list): a list of entity mentions.
    :param token_num (int): the number of tokens.
    :return:a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    count = 0
    for entity in entities:
        start, end = entity["start"], entity["end"]
        if end > token_num:
            continue
        entity_type = entity["entity_type"]
        if any([labels[i] != 'O' for i in range(start, end)]):
            count += 1
            continue
        labels[start] = 'B-{}'.format(entity_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(entity_type)
    if count:
        print('cannot cover {} entities due to span overlapping'.format(count))
    return labels

class IEDataset(Dataset):
    def __init__(self, path, max_length, use_type):
        self.path = path
        self.data = []
        self.max_length = max_length
        self.use_type = use_type
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for entity in inst['entities']:
                type_set.add(entity["entity_type"])
        return type_set

    @property
    def token_set(self):
        type_set = set()
        for inst in self.data:
            for token in inst['tokens']:
                type_set.add(token)
        return type_set

    @property
    def char_set(self):
        type_set = set()
        for inst in self.data:
            for token in inst['chars']:
                for char in token:
                    type_set.add(char)
        return type_set

    def load_data(self):
        """Load data from file."""
        raw_line = []
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['tokens'])
            if inst_len > self.max_length:
                print("over max token length with length {}".format(inst_len))
                continue
            insts.append(inst)
        self.preprocess(insts)
        print('Loaded {} instances from {}'.format(len(self), self.path))

    def preprocess(self, data):
        for d in data:
            # filter class
            entities = []
            for ent in d['entity_mentions']:
                if ent['entity_type'] in self.use_type:
                    entities.append(ent)
            self.data.append({
                'tokens': d['tokens'],
                'chars': [[c for c in t] for t in d['tokens']],
                'entities': entities
            })

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        entity_type_stoi = vocabs.get('entity_type', None)
        entity_label_stoi = vocabs.get('entity_label', None)
        word_stoi = vocabs.get('word2idx', None)
        char_stoi = vocabs.get('char2idx', None)

        data = []
        skip_num = 0
        for inst in self.data:
            tokens = inst['tokens']
            entities = inst['entities']
            entities.sort(key=lambda x: x["start"])
            token_num = len(tokens)

            # bert tokenization
            pieces = [tokenizer.tokenize(t) for t in tokens]
            token_lens = [len(x) for x in pieces]
            if 0 in token_lens:
                skip_num += 1
                continue
            pieces = [p for ps in pieces for p in ps]
            if len(pieces) == 0:
                skip_num += 1
                continue
            
            # Pad word pieces with special tokens
            piece_idxs = tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
            
            if sum(token_lens) < self.max_length -2:
                assert sum(token_lens) +2 == len(piece_idxs)
            else:
                print('truncated when creating pieces with {}!!!'.format(sum(token_lens)))
            
            pad_num = self.max_length - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            pad_id = tokenizer.encode(['<pad>'], add_special_tokens=False)[0]
            piece_idxs = piece_idxs + [pad_id] * pad_num

            # word embedding
            words = [word_stoi[w] if w in word_stoi else word_stoi['<UNK>'] for w in inst['tokens']]

            # char embedding
            chars = [[char_stoi[w] if w in char_stoi else char_stoi['<UNK>']for w in t] for t in inst['chars']]

            
            entity_labels = get_entity_labels(entities, token_num)
            entity_label_idxs = [entity_label_stoi[l] for l in entity_labels]
            entity_list = [[e["start"], e["end"], e["entity_type"]]
                           for e in entities if e["end"] <= token_num]
            
            instance = Instance(
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask,
                entity_label_idxs=entity_label_idxs,
                entity_list=entity_list,
                word_idxs=words,
                char_idxs=chars
            )
            data.append(instance)
        if skip_num:
            print("skip {} instances".format(skip_num))
        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_entity_labels = []
        batch_token_lens = []
        batch_attention_masks = []
        batch_entity_lists = []
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        char_nums = [torch.cuda.LongTensor([len(t) for t in inst.char_idxs]) for inst in batch] # a list of list of integer
        batch_word_idxs = []
        batch_char_idxs = [] # list of 2D tensor

        for j, inst in enumerate(batch):
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_tokens.append(inst.tokens)
            batch_entity_lists.append(inst.entity_list)
            
            # for identification
            batch_entity_labels.append(inst.entity_label_idxs +
                                       [0] * (max_token_num - token_num))
            # word embedding
            batch_word_idxs.append(inst.word_idxs +
                    [0] * (max_token_num - token_num)) # 0 is pad
            # char embedding
            char_idxs = [] # #of_token * max_char_len
            max_subchar_num = max(char_nums[j])
            for char_list in inst.char_idxs:
                char_idxs.append(char_list + [0] * (max_subchar_num-len(char_list))) # 0 is pad
            batch_char_idxs.append(torch.cuda.LongTensor(char_idxs))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)

        batch_entity_labels = torch.cuda.LongTensor(batch_entity_labels)
        token_nums = torch.cuda.LongTensor(token_nums)

        batch_word_idxs = torch.cuda.LongTensor(batch_word_idxs)

        return Batch(
            tokens=batch_tokens,
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            entity_label_idxs=batch_entity_labels,
            entity_lists=batch_entity_lists,
            token_nums=token_nums,
            word_idxs=batch_word_idxs,
            char_idxs=batch_char_idxs,
            char_nums=char_nums
        )
