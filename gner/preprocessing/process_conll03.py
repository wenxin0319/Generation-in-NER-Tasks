from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
import time


# set up some global variables
inv_map = {"train": 0, "dev": 1, "test": 2}
ord_map = ["train", "dev", "test"]
sent_num = [0 for i in range(3)]
sentences = [[] for i in range(3)]
total_tokens = [0 for i in range(3)]
doc_sentence_start = [[] for i in range(3)]
sent_starts = [[] for i in range(3)]
sent_length = [[] for i in range(3)]
entities = [[] for i in range(3)]

class Conll03Reader:
    def read(self, data_path):
        data_parts = ['train', 'dev', 'test']
        extension = '.txt'
        dataset = {}
        for data_part in tqdm(data_parts):
            file_path = os.path.join(data_path, data_part+extension)
            dataset[data_part] = self.read_file(str(file_path), str(data_part))
        return dataset

    def read_file(self, file_path, data_part):
        samples = []
        tokens = []
        tags = []
        with open(file_path, 'r',encoding='utf-8') as fb:
            for line in fb:
                line = line.strip('\n')

                if line == '-DOCSTART- -X- -X- O':
                    pass
                elif line =='':
                    if len(tokens) != 0:
                        samples.append((tokens, tags))
                        tokens = []
                        tags = []
                else:
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    tags.append(contents[-1])

        sent_num[inv_map[data_part]] = len(samples)
        sentences[inv_map[data_part]] = [_sentence[0] for _sentence in samples] # the first element of a tuple
        sent_length[inv_map[data_part]] = [len(sent) for sent in sentences[inv_map[data_part]] ]
        cnt = 0
        for _ in range(sent_num[inv_map[data_part]]):
            doc_sentence_start[inv_map[data_part]].append(cnt)
            cnt = cnt + sent_length[inv_map[data_part]][_]
        total_tokens[inv_map[data_part]] = sum([len(sent) for sent in sentences[inv_map[data_part]] ])
        doc_sentence_start[inv_map[data_part]].append(total_tokens)
        sent_starts[inv_map[data_part]] = doc_sentence_start[inv_map[data_part]] + [total_tokens[inv_map[data_part]]]
        
        def process_sentence(_sentence_index):  
            ret_entity_list = []
            tmp_entity = ""
            tmp_start = sent_starts[inv_map[data_part]][_sentence_index]
            tmp_end = sent_starts[inv_map[data_part]][_sentence_index]
            for _word_index in range(sent_length[inv_map[data_part]][_sentence_index]):
                #add some condition
                if samples[_sentence_index][1][_word_index][0] == 'B':
                    if(_word_index!=0):
                        tmp_end = sent_starts[inv_map[data_part]][_sentence_index] + _word_index - 1
                        ret_entity_list.append((tmp_start, tmp_end, tmp_entity))
                    tmp_entity = samples[_sentence_index][1][_word_index][2:]
                    tmp_start = sent_starts[inv_map[data_part]][_sentence_index] + _word_index 
                if samples[_sentence_index][1][_word_index][0] == 'O' :
                        tmp_end = sent_starts[inv_map[data_part]][_sentence_index] + _word_index - 1
                        ret_entity_list.append((tmp_start, tmp_end, tmp_entity))
                        tmp_entity = samples[_sentence_index][1][_word_index]
                        tmp_start = sent_starts[inv_map[data_part]][_sentence_index] + _word_index                   
                if _word_index == sent_length[inv_map[data_part]][_sentence_index]-1:
                    tmp_end = sent_starts[inv_map[data_part]][_sentence_index] + _word_index
                    ret_entity_list.append((tmp_start, tmp_end, tmp_entity))    
                else:
                    pass
            ret_entity_list.append((tmp_start, tmp_end, tmp_entity))

            return ret_entity_list
        
        entities[inv_map[data_part]] = [process_sentence(_sentence_index) for _sentence_index in range(len(samples))]
        
        return samples

def convert(data, output_path, tokenizer):
    dic_type={}
    choice = ["train", "test", "dev"]
    for choose in choice:
        print(choose)
        with open(str(output_path) + str(choose) + ".json", 'w', encoding='utf-8') as w:
            doc_id = choose 
            window_size = 1
                
            offset = 0

            for i in range(sent_num[inv_map[choose]] - window_size + 1):
                wnd_tokens = data[choose][i][0] # the first element of a tuple
                wnd_relations = []
                wnd_events = []
                def slice_fn(lst, ind, wnd): 
                    return [item for j in range(wnd) for item in lst[ind+j]]
                wnd_entities = [slice_fn(
                    lst, i, window_size) for lst in [entities[inv_map[choose]]] ]

                wnd_sent_starts = sent_starts[inv_map[choose]][i:i+window_size+1]
                wnd_start, wnd_end = wnd_sent_starts[0], wnd_sent_starts[-1]

                wnd_id = '{}-{}'.format(doc_id, i)
                pieces = [tokenizer.tokenize(t) for t in wnd_tokens]
                word_lens = [len(p) for p in pieces]            

                wnd_entities_ = []
                wnd_entity_map = {}
  
                cnt = 0
                for j, (start, end, entity_type) in enumerate(wnd_entities[0]):
                    start, end = start - offset, end - offset + 1
                    entity_id = '{}-E{}'.format(wnd_id, cnt)
                    text=' '.join(wnd_tokens[start:end])
                    if entity_type == 'O':
                        continue
                    cnt += 1
                    if(len(text)>0):
                        if entity_type not in dic_type:
                            dic_type[entity_type]=0
                        else:
                            dic_type[entity_type]=dic_type[entity_type]+1
                        entity = {
                            'id': entity_id,
                            'start': start, 
                            'end': end,
                            'entity_type': entity_type,
                            'text': text}
                        wnd_entities_.append(entity)
                        wnd_entity_map[(start, end)] = entity

                wnd_ = {
                    'doc_id': doc_id,
                    'wnd_id': wnd_id,
                    'entity_mentions': wnd_entities_,
                    'tokens': wnd_tokens,
                    'pieces': [p for w in pieces for p in w],
                    'token_lens': word_lens,
                    'sentence': ' '.join(wnd_tokens),               
                }
                w.write(json.dumps(wnd_) + '\n')
                offset += len(sentences[inv_map[choose]][i])
    with open (os.path.join(output_path,"type.json"), 'w', encoding='utf-8') as w1:
        w1.write(json.dumps(dic_type))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-b', '--bert', help='BERT model name', default='bert-large-cased')
    parser.add_argument('-i', '--input', help='Input CoNLL folder path', required=True)
    parser.add_argument('-o', '--output', help='Path to the output folder')
    args = parser.parse_args()
    model_name = args.bert
    if model_name.startswith('bert-'):
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    elif model_name.startswith('roberta-'):
        bert_tokenizer = RobertaTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, do_lower_case=False, do_fast=False)
    
    ds_rd = Conll03Reader()
    data = ds_rd.read(args.input)
    output_path = args.output
    convert(data, output_path, bert_tokenizer)
