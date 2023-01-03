from tensorboardX import SummaryWriter

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_vocabs(datasets):
    event_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        event_type_set.update(dataset.event_type_set)
        role_type_set.update(dataset.role_type_set)
    
    event_type_itos = sorted(event_type_set)
    role_type_itos = sorted(role_type_set)
    
    event_type_stoi = {k: i for i, k in enumerate(event_type_itos)}
    role_type_stoi = {k: i for i, k in enumerate(role_type_itos)}
    
    return {
        'event_type_itos': event_type_itos,
        'event_type_stoi': event_type_stoi,
        'role_type_itos': role_type_itos,
        'role_type_stoi': role_type_stoi,
    }

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

def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None, no_bos=False):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.

    If we use T5, which is a tokenizer do not have [SOS], please set "no_bos" as True.
    """
    words = []
    for s in span.split(' '):
        if not no_bos:
            words.extend(tokenizer.encode(s, add_special_tokens=True)[1:-1]) # ignore [SOS] and [EOS]
        else:
            words.extend(tokenizer.encode(s, add_special_tokens=True)[:-1]) # ignore [SOS] and [EOS]
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]