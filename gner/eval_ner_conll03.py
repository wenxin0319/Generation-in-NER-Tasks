import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, T5Tokenizer
from model import GenerativeModel
from data_ner_conll import NERCoNLL_Dataset
from utils import compute_f1, get_span_idx
from argparse import ArgumentParser, Namespace
from template import NERCoNLL_template
import ipdb

# This scorer need to be adjust based on the task.
def cal_scores(gold_triggers, pred_triggers):
    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set([(t['start'], t['end']) for t in gold_trigger])
        pred_set = set([(t[0], t[1]) for t in pred_trigger])
        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)
    
    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set([(t['start'], t['end'], t['entity_type']) for t in gold_trigger])
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)
    

    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
    }
    
    return scores

# configuration
parser = ArgumentParser()
parser.add_argument('-cner', '--ner_config', required=True)
parser.add_argument('-ner', '--ner_model', required=True)
parser.add_argument('--no_dev', action='store_true', default=False)
parser.add_argument('--write_result', action='store_true', default=False)
args = parser.parse_args()

# load config
with open(args.ner_config) as fp:
    ner_config = json.load(fp)
ner_config = Namespace(**ner_config)

# fix random seed
np.random.seed(ner_config.seed)
torch.manual_seed(ner_config.seed)
torch.backends.cudnn.enabled = False

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)

# set GPU device
torch.cuda.set_device(ner_config.gpu_device)

# check valid styles
assert np.all([style in ['task_def', 'template'] for style in ner_config.input_style])

# output
output_dir = '/'.join(args.ner_model.split('/')[0:-1])
dev_prediction_path = os.path.join(output_dir, 'pred.dev.final.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.final.json')

# tokenizer
no_bos = False
if ner_config.model_name.startswith("facebook/bart-"):
    tokenizer = BartTokenizer.from_pretrained(ner_config.model_name, cache_dir=ner_config.cache_dir)
elif ner_config.model_name.startswith("copy+facebook/bart-"):
    model_name = ner_config.model_name.split('copy+', 1)[1]
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=ner_config.cache_dir)
elif ner_config.model_name.startswith("t5-"):
    tokenizer = T5Tokenizer.from_pretrained(ner_config.model_name, cache_dir=ner_config.cache_dir)
    no_bos=True
elif ner_config.model_name.startswith("copy+t5-"):
    model_name = ner_config.model_name.split('copy+', 1)[1]
    no_bos=True
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=ner_config.cache_dir)
else:
    raise ValueError('The model name that specified is invalid for our code.')
special_tokens = ner_config.special_tokens
tokenizer.add_tokens(special_tokens)

if hasattr(ner_config, 'shuffle_class_order'):
    ner_config.shuffle_class_order = False

# load data
dev_set = NERCoNLL_Dataset(ner_config.input_style, tokenizer, ner_config.dev_file, ner_config.test_class_options, ner_config.max_token_length, ner_config.max_input_length, ner_config.max_output_length, 0.0, no_bos, ner_config.shuffle_class_order)
test_set = NERCoNLL_Dataset(ner_config.input_style, tokenizer, ner_config.test_file, ner_config.test_class_options, ner_config.max_token_length, ner_config.max_input_length, ner_config.max_output_length, 0.0, no_bos, ner_config.shuffle_class_order)

dev_batch_num = len(dev_set) // ner_config.eval_batch_size + (len(dev_set) % ner_config.eval_batch_size != 0)
test_batch_num = len(test_set) // ner_config.eval_batch_size + (len(test_set) % ner_config.eval_batch_size != 0)

# load model
logger.info(f"Loading model from {args.ner_model}")
ner_model = GenerativeModel(ner_config, tokenizer)
ner_model.load_state_dict(torch.load(args.ner_model, map_location=f'cuda:{ner_config.gpu_device}'))
ner_model.cuda(device=ner_config.gpu_device)
ner_model.eval()

# eval dev set
if not args.no_dev:
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
    ner_model.eval()
    write_output = []
    dev_gold_triggers, dev_pred_triggers = [], []
    
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=ner_config.eval_batch_size, 
                                                 shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        pred_text = ner_model.predict(batch, ner_config.beam_size, max_length=ner_config.max_output_length)
        gold_text = batch.target_text
        input_text = batch.input_text
        ground_truths = batch.ground_truth
        original_tokens = batch.tokens
        for bid, (i_text, g_text, p_text, gold, ori_tokens) in enumerate(zip(input_text, gold_text, pred_text, ground_truths, original_tokens)):
            template = NERCoNLL_template(ner_config.input_style, ori_tokens, ner_config.test_class_options)
            
            # decode predictions
            pred_object = template.decode(p_text)
            
            # map prediction back to span prediction
            pred_span = [get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], mention, tokenizer, None, no_bos)+(pred_type, ) for mention, pred_type in pred_object]

            pred_span = [p for p in pred_span if p[0] != -1]

            # collect things
            dev_pred_triggers.append(pred_span)
            dev_gold_triggers.append(gold)
            write_output.append({
                'input text': i_text, 
                'gold text': g_text,
                'pred text': p_text,
                'gold object': gold,
                'pred object': pred_span,
                'tokens': ori_tokens
            })
    progress.close()
    
    dev_scores = cal_scores(dev_gold_triggers, dev_pred_triggers)

    # print scores
    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_id'][3] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][1], 
        dev_scores['tri_id'][4] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][0], dev_scores['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_cls'][3] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][1], 
        dev_scores['tri_cls'][4] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][0], dev_scores['tri_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")

    if args.write_result:
        with open(dev_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)
    
    
# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
ner_model.eval()
write_output = []
test_gold_triggers, test_pred_triggers = [], []

for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=ner_config.eval_batch_size, 
                                            shuffle=False, collate_fn=test_set.collate_fn)):
    progress.update(1)
    #pred_text = ner_model.predict(batch, ner_config.beam_size, max_length=ner_config.max_output_length, desinated_prefix='location : ')
    pred_text = ner_model.predict(batch, ner_config.beam_size, max_length=ner_config.max_output_length)
    gold_text = batch.target_text
    input_text = batch.input_text
    ground_truths = batch.ground_truth
    original_tokens = batch.tokens
    for bid, (i_text, g_text, p_text, gold, ori_tokens) in enumerate(zip(input_text, gold_text, pred_text, ground_truths, original_tokens)):
        template = NERCoNLL_template(ner_config.input_style, ori_tokens, ner_config.test_class_options)
        
        # decode predictions
        pred_object = template.decode(p_text)
        #pred_object = template.decode(g_text)

        # map prediction back to span prediction
        pred_span = [get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], mention, tokenizer, None, no_bos)+(pred_type, ) for mention, pred_type in pred_object]

        pred_span = [p for p in pred_span if p[0] != -1]
        
        test_pred_triggers.append(pred_span)
        test_gold_triggers.append(gold)
        write_output.append({
            'input text': i_text, 
            'gold text': g_text,
            'pred text': p_text,
            'gold object': gold,
            'pred object': pred_span,
            'tokens': ori_tokens
            #'input_tokens': tokenizer.convert_ids_to_tokens(batch.enc_idxs[bid])
        })

progress.close()

test_scores = cal_scores(test_gold_triggers, test_pred_triggers)

# print scores
print("---------------------------------------------------------------------")
print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
    test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
    test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
print("---------------------------------------------------------------------")

if args.write_result:
    with open(test_prediction_path, 'w') as fp:
        json.dump(write_output, fp, indent=4)