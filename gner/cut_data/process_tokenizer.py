import json
import os
from transformers import (BertTokenizer, AdamW,
                          RobertaTokenizer,
                          XLMRobertaTokenizer,
                          BartTokenizer,
                          MBart50Tokenizer,
                          MT5Tokenizer,
                          AutoTokenizer,
                          T5Tokenizer,
                          get_linear_schedule_with_warmup)

def process_different_tokenizer(name_list,output_path,model_name):

    if model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(model_name,cache_dir='./cache')

    elif model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name,cache_dir='./cache')

    elif  model_name.startswith('t5-'):
        tokenizer = T5Tokenizer.from_pretrained(model_name,cache_dir='./cache')

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='./cache')

    for i in name_list:
        type=i.split("/")[2]
        with open(i + ".json",'r',encoding='utf-8') as r:
            extension = '.json'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_path = os.path.join(output_path, str(type)+extension)

            with open(file_path,'w',encoding='utf-8') as w:
                r_read = r.readlines()
                for r__ in r_read:
                    answer = json.loads(r__)
                    sentences= answer['sentence'].split()
                    pieces1 = [tokenizer.tokenize(t) for t in sentences]
                    word_lens = [len(p) for p in pieces1] 
                    answer['tokens']=sentences
                    answer['pieces']=[p for w in pieces1 for p in w]
                    answer['token_lens']=word_lens
                    w.write(json.dumps(answer) + "\n") 
          