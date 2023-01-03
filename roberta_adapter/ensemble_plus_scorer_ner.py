import ipdb
from argparse import ArgumentParser
from collections import defaultdict
import json
from pprint import pprint

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

class score_by_class(object):
    def __init__(self, gold_trigger_cls_num=0, pred_trigger_cls_num=0, trigger_cls_num=0):
        self.gold_trigger_cls_num=gold_trigger_cls_num
        self.pred_trigger_cls_num=pred_trigger_cls_num
        self.trigger_cls_num=trigger_cls_num

    def get_score(self):
        trigger_prec, trigger_rec, trigger_f = compute_f1(
            self.pred_trigger_cls_num, self.gold_trigger_cls_num, self.trigger_cls_num)
        
        scores = {
            'entities': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
            'pred_num': self.pred_trigger_cls_num,
            'gold_num': self.gold_trigger_cls_num,
            'match_num': self.trigger_cls_num
        }
        return scores

def score_graphs(gold_graphs, pred_graphs):
    gold_ent_num = pred_ent_num = ent_match_num = 0

    detailed_scores = defaultdict(score_by_class)

    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
        # Entity
        gold_entities = [(e[0], e[1], e[2]) for e in gold_graph['entities']]
        gold_entities = list(set(gold_entities))
        gold_entities = [[e[0], e[1], e[2]] for e in gold_entities]
        pred_entities = [(e[0], e[1], e[2]) for e in pred_graph['entities']]
        pred_entities = list(set(pred_entities))
        pred_entities = [[e[0], e[1], e[2]] for e in pred_entities]

        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])

        for pred_ent in pred_entities:
            detailed_scores[pred_ent[2]].pred_trigger_cls_num += 1
            matched = [item for item in gold_entities
                       if item[0] == pred_ent[0] and item[1] == pred_ent[1] and item[2] == pred_ent[2]]
            if matched:
                detailed_scores[pred_ent[2]].trigger_cls_num += 1
        
        for gold_ent in gold_entities:
            detailed_scores[gold_ent[2]].gold_trigger_cls_num += 1

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    
    print("---------------------------------------------------------------------")
    print('Entity     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        entity_prec * 100.0, ent_match_num, pred_ent_num, 
        entity_rec * 100.0, ent_match_num, gold_ent_num, entity_f * 100.0))
    
    print("---------------------------------------------------------------------")
    
    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f},
    }
    scores_by_type = dict()
    for k, v in detailed_scores.items():
        scores_by_type[k] = v.get_score()

    return scores, scores_by_type

def transform(gold_file_list):
    gold_json = {}
    for g in gold_file_list:
        entities = [[e['start'], e['end'], e['entity_type']] for e in g['entity_mentions']]

        #if ' '.join(g['tokens']) in gold_json.keys():
        #    print(' '.join(g['tokens']))

        gold_json[' '.join(g['tokens'])]={
            "entities": entities,
            "tokens": g['tokens']
        }
    return gold_json

def transform_pred(pred_file_lists):
    pred_json = {}
    for f_pred in pred_file_lists:
        preds = json.load(open(f_pred, 'r'))
        for g in preds:
            entities = g['pred object']
            
            if ' '.join(g['tokens']) not in pred_json.keys():         
                pred_json[' '.join(g['tokens'])]={
                    "entities": entities,
                    "tokens": g['tokens']
                }
            else:
                pred_json[' '.join(g['tokens'])]['entities'].extend(entities)
    return pred_json

# configuration
parser = ArgumentParser()
parser.add_argument('-p', '--pred_path', required=True, nargs='+', help='The prediction files.')
parser.add_argument('-g', '--gold_path', required=True, help='The test file.')
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()

golds = [json.loads(line) for line in open(args.gold_path, 'r')]
golds = transform(golds)
#predictions = transform_pred(json.load(open(args.pred_path, 'r')))
predictions = transform_pred(args.pred_path)

print(len(golds.keys()))
assert len(golds.keys()) == len(predictions.keys())

pred_graphs = []
gold_graphs = []
for key, pred_graph in predictions.items():
    gold_graph = golds[key]
    pred_graphs.append(pred_graph)
    gold_graphs.append(gold_graph)

full_scores, scores_by_type = score_graphs(gold_graphs, pred_graphs)

if args.verbose:
    pprint(scores_by_type)