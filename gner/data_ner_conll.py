import json, logging, pickle
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from template import NERCoNLL_template
import ipdb
import random

logger = logging.getLogger(__name__)

ner_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'entities', 'valid_types']
NERInstance = namedtuple('NERInstance', field_names=ner_instance_fields, defaults=[None] * len(ner_instance_fields))

ner_batch_fields = ['wnd_ids', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'token_start_idxs', 'ground_truth', 'input_text', 'target_text', 'enc_idxs', 'enc_attn', 'dec_idxs', 'dec_attn', 'lbl_idxs', 'raw_lbl_idxs']

NERBatch = namedtuple('NERBatch', field_names=ner_batch_fields, defaults=[None] * len(ner_batch_fields))

def filter_by_class(entities, used_class):
    entities_ = []
    for e in entities:
        if e['entity_type'] not in used_class:
            pass
        else:
            entities_.append(e)
    return entities_

class NERCoNLL_Dataset(Dataset):
    def __init__(self, input_style, tokenizer, path, used_class, max_token_length=128, max_input_length=300, 
    max_output_length=50, class_dropout=0.0, no_bos=False, shuffle_class_order=True, duplicate_out_template=False):
        self.input_style = input_style
        self.tokenizer = tokenizer
        self.path = path
        self.data = []
        self.insts = []
        self.max_token_length = max_token_length
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.used_class = used_class
        self.class_dropout = class_dropout
        self.no_bos = no_bos # if you use bart, then this should be False; if you use t5, then this should be True
        self.shuffle_class_order = shuffle_class_order
        self.duplicate_out_template = duplicate_out_template
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_token_length:
                print("over max token length with length {}".format(inst_len))
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            entities = inst['entity_mentions']
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            piece_idxs = self.tokenizer.encode(pieces, add_special_tokens=True, max_length=self.max_token_length+2, truncation=True)
            piece_idxs = piece_idxs[:-1] if self.no_bos else piece_idxs[1:-1]
            assert sum(token_lens) == len(piece_idxs)
                
            token_start_idxs = [sum(token_lens[:_]) for _ in range(len(token_lens))] + [sum(token_lens)]

            instance = NERInstance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                token_start_idxs=token_start_idxs,
                entities=entities,
                valid_types=inst['valid_types']
            )
            self.data.append(instance)
            
        logger.info(f'Loaded {len(self)}/{len(lines)} instances from {self.path}')

    def collate_fn(self, batch):
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        piece_idxs = [inst.piece_idxs for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        token_start_idxs = [inst.token_start_idxs for inst in batch]

        ground_truth = []
        input_text = []
        target_text = []
        for inst in batch:
            ####
            # Here, we have a random process that decide for each instance, whether we would like to dropout some class
            ####
            used_class_this = []
            for c in self.used_class:
                if c not in inst.valid_types:
                    continue
                if random.random() >= self.class_dropout:
                    used_class_this.append(c)

            if self.shuffle_class_order:
                random.shuffle(used_class_this)
            ####
            # Based on the used class for this batch, we create the training example
            ####
            
            filtered_entity = filter_by_class(inst.entities, used_class_this)
            template_obj = NERCoNLL_template(self.input_style, inst.tokens, class_options=used_class_this, ground_truth=filtered_entity)
            input_text_ = template_obj.generate_input_str(None)
            target_text_, _ = template_obj.generate_output_str(None)            
            ground_truth.append(filtered_entity)
            input_text.append(input_text_)
            if self.duplicate_out_template:
                target_text.append(template_obj.output_template(None) + " [SEP] " + target_text_)
            else:
                target_text.append(target_text_)
        
        ####
        # This is checking code, can be commented out
        ####
        inputs_ = self.tokenizer(input_text, padding='longest')
        if len(inputs_['input_ids'][0]) > self.max_input_length-2:
            logger.info('Input Length with {} pieces is longer that we set'.format(len(inputs_['input_ids'][0])))
        target_ = self.tokenizer(target_text, padding='longest')
        if len(target_['input_ids'][0]) > self.max_output_length-2:
            logger.info('Target Length with {} pieces is longer that we set'.format(len(target_['input_ids'][0])))
        ####
        # This is the end of checking code
        ####       

        # encoder inputs
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.max_input_length, truncation=True)
        enc_idxs = inputs['input_ids']
        enc_attn = inputs['attention_mask']

        # decoder inputs
        targets = self.tokenizer(target_text, return_tensors='pt', padding=True, max_length=self.max_output_length, truncation=True)
        dec_idxs = targets['input_ids']
        batch_size = dec_idxs.size(0)
        if self.no_bos:
            # This is for T5
            tt = torch.ones((batch_size, 1), dtype=torch.long)
            tt[:] = self.tokenizer.pad_token_id # <pad> as a start token
            dec_idxs = torch.cat((tt, dec_idxs), dim=1)
            dec_attn = torch.cat((torch.ones((batch_size, 1), dtype=torch.long), targets['attention_mask']), dim=1)
        else:
            dec_idxs[:, 0] = self.tokenizer.eos_token_id # This is for BART
            dec_attn = targets['attention_mask']
            
        # labels
        padding = torch.ones((batch_size, 1), dtype=torch.long)
        padding[:] = self.tokenizer.pad_token_id
        raw_lbl_idxs = torch.cat((dec_idxs[:, 1:], padding), dim=1)
        lbl_attn = torch.cat((dec_attn[:, 1:], torch.zeros((batch_size, 1), dtype=torch.long)), dim=1)
        lbl_idxs = raw_lbl_idxs.masked_fill(lbl_attn==0, -100) # ignore padding
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        dec_idxs = dec_idxs.cuda()
        dec_attn = dec_attn.cuda()
        raw_lbl_idxs = raw_lbl_idxs.cuda()
        lbl_idxs = lbl_idxs.cuda()        

        return NERBatch(
            wnd_ids=wnd_ids,
            tokens=tokens,
            pieces=pieces,
            piece_idxs=piece_idxs,
            token_lens=token_lens,
            token_start_idxs=token_start_idxs,
            ground_truth=ground_truth,
            input_text=input_text,
            target_text=target_text,
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            dec_idxs=dec_idxs,
            dec_attn=dec_attn,
            lbl_idxs=lbl_idxs,
            raw_lbl_idxs=raw_lbl_idxs
        )
