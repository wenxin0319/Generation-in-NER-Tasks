import os, sys, json, logging, time, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from model import GenerativeModel
from data_ner_conll import NERCoNLL_Dataset
from utils import Summarizer, compute_f1, get_span_idx
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
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# set GPU device
torch.cuda.set_device(config.gpu_device)

# check valid styles
assert np.all([style in ['task_def', 'template'] for style in config.input_style])

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

# tokenizer
no_bos = False
if config.model_name.startswith("facebook/bart-"):
    tokenizer = BartTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("copy+facebook/bart-"):
    model_name = config.model_name.split('copy+', 1)[1]
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
elif config.model_name.startswith("t5-"):
    tokenizer = T5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    no_bos=True
elif config.model_name.startswith("copy+t5-"):
    model_name = config.model_name.split('copy+', 1)[1]
    no_bos=True
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
else:
    raise ValueError('The model name that specified is invalid for our code.')

special_tokens = config.special_tokens
tokenizer.add_tokens(special_tokens)

# load data
train_set = NERCoNLL_Dataset(config.input_style, tokenizer, config.train_file, config.used_class, config.max_token_length, config.max_input_length, config.max_output_length, config.class_dropout, no_bos, config.shuffle_class_order)
#dev_set = NERCoNLL_Dataset(config.input_style, tokenizer, config.dev_file, config.test_class_options, config.max_token_length, config.max_input_length, config.max_output_length, 0.0, no_bos, config.shuffle_class_order)
dev_set = NERCoNLL_Dataset(config.input_style, tokenizer, config.dev_file, config.used_class, config.max_token_length, config.max_input_length, config.max_output_length, 0.0, no_bos, config.shuffle_class_order)
test_set = NERCoNLL_Dataset(config.input_style, tokenizer, config.test_file, config.test_class_options, config.max_token_length, config.max_input_length, config.max_output_length, 0.0, no_bos, config.shuffle_class_order)

train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = GenerativeModel(config, tokenizer)
model.cuda(device=config.gpu_device)
# optimizer
#param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]

param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('model.linear_copy')],
        'lr': 1e-3, 'weight_decay': 1e-6
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('model.linear_copy')],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    }
]

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num*config.warmup_epoch,
                                           num_training_steps=train_batch_num*config.max_epoch)


# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'tri_id': (0.0, 0.0, -1),
    'tri_cls': (0.0, 0.0, -1)
}
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step, 
                                                 shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        
        # forard model
        loss = model(batch)
        
        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        
        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()

    # eval dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    best_dev_flag = False
    write_output = []
    dev_gold_triggers, dev_pred_triggers = [], []
    
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size, 
                                                 shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        pred_text = model.predict(batch, config.beam_size, max_length=config.max_output_length)
        gold_text = batch.target_text
        input_text = batch.input_text
        ground_truths = batch.ground_truth
        original_tokens = batch.tokens
        for bid, (i_text, g_text, p_text, gold, ori_tokens) in enumerate(zip(input_text, gold_text, pred_text, ground_truths, original_tokens)):
            #template = NERCoNLL_template(config.input_style, ori_tokens, config.test_class_options)
            template = NERCoNLL_template(config.input_style, ori_tokens, config.used_class)
            
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
    
    # check best dev model
    if dev_scores['tri_cls'][5] > best_dev_scores['tri_cls'][2]:
        best_dev_flag = True
    

    # if best dev, save model and evaluate test set
    if best_dev_flag:    
        best_dev_scores = {
            'tri_id': (dev_scores['tri_id'][3], dev_scores['tri_id'][4], dev_scores['tri_id'][5]),
            'tri_cls': (dev_scores['tri_cls'][3], dev_scores['tri_cls'][4], dev_scores['tri_cls'][5])
        }
        best_dev_epoch = epoch
        
        # save best model
        logger.info('Saving best model')
        torch.save(model.state_dict(), best_model_path)
        
        # save dev result
        with open(dev_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)

        # eval test set
        progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
        model.eval()
        write_output = []
        test_gold_triggers, test_pred_triggers = [], []
        
        for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, 
                                                    shuffle=False, collate_fn=test_set.collate_fn)):
            progress.update(1)
            pred_text = model.predict(batch, config.beam_size, max_length=config.max_output_length)
            gold_text = batch.target_text
            input_text = batch.input_text
            ground_truths = batch.ground_truth
            original_tokens = batch.tokens
            for bid, (i_text, g_text, p_text, gold, ori_tokens) in enumerate(zip(input_text, gold_text, pred_text, ground_truths, original_tokens)):
                template = NERCoNLL_template(config.input_style, ori_tokens, config.test_class_options)
                
                # decode predictions
                pred_object = template.decode(p_text)
                
                # map prediction back to span prediction
                pred_span = [get_span_idx(batch.piece_idxs[bid], batch.token_start_idxs[bid], mention, tokenizer, None, no_bos)+(pred_type, ) for mention, pred_type in pred_object]
                pred_span = [p for p in pred_span if p[0] != -1]

                # collect things
                test_pred_triggers.append(pred_span)
                test_gold_triggers.append(gold)
                write_output.append({
                    'input text': i_text, 
                    'gold text': g_text,
                    'pred text': p_text,
                    'gold object': gold,
                    'pred object': pred_span,
                    'tokens': ori_tokens
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

        # save test result
        with open(test_prediction_path, 'w') as fp:
            json.dump(write_output, fp, indent=4)
            
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    if best_dev_flag:
        logger.info({"epoch": epoch, "test_scores": test_scores})
    logger.info("Current best")
    logger.info({"best_epoch": best_dev_epoch, "best_scores": best_dev_scores})
        
logger.info(log_path)
logger.info("Done!")

