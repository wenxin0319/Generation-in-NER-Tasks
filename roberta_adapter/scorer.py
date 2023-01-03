from collections import defaultdict

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
    def __init__(self, gold_ent_cls_num=0, pred_ent_cls_num=0, ent_cls_num=0):
        self.gold_ent_cls_num=gold_ent_cls_num
        self.pred_ent_cls_num=pred_ent_cls_num
        self.ent_cls_num=ent_cls_num

    def get_score(self):
        return compute_f1(self.pred_ent_cls_num, self.gold_ent_cls_num, self.ent_cls_num)

    def __str__(self):
        print('gold number: {} \t pred_number: {} \t match_number: {}'.format(self.gold_ent_cls_num, self.pred_ent_cls_num, self.ent_cls_num))

def score_graphs(gold_ents, pred_ents):
    gold_ent_num = pred_ent_num = ent_match_num = 0
    detailed_scores = defaultdict(score_by_class)

    for gs, ps in zip(gold_ents, pred_ents):
        # Entity
        gold_entities = [(e[0], e[1], e[2]) for e in gs]
        gold_entities = list(set(gold_entities))
        gold_entities = [[e[0], e[1], e[2]] for e in gold_entities]
        for g in gold_entities:
            detailed_scores[g[2]].gold_ent_cls_num+=1

        pred_entities = [(e[0], e[1], e[2]) for e in ps]
        pred_entities = list(set(pred_entities))
        pred_entities = [[e[0], e[1], e[2]] for e in pred_entities]
        for p in pred_entities:
            detailed_scores[p[2]].pred_ent_cls_num+=1

        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        for entity in pred_entities:
            if entity in gold_entities:
                ent_match_num += 1
                detailed_scores[entity[2]].ent_cls_num+=1

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    
    print("---------------------------------------------------------------------")
    print('Entity     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        entity_prec * 100.0, ent_match_num, pred_ent_num, 
        entity_rec * 100.0, ent_match_num, gold_ent_num, entity_f * 100.0))
    
    scores_by_type = dict()
    for k, v in detailed_scores.items():
        scores_by_type[k] = v.get_score()

    return entity_prec, entity_rec, entity_f, detailed_scores
