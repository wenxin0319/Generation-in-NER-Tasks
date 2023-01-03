import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel, RobertaModel, XLMRobertaModel, BartModel, AutoModel
from collections import Counter
import copy
import ipdb

def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask

def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets

def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

def tag_paths_to_spans(paths, token_nums, vocab, type_vocab):
    """
    Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)
            #tag = type_vocab[tag]
            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        batch_mentions.append(mentions)
    return batch_mentions

class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        # self.same_type = self.map_same_types()
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens):
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores


class StructuralModel(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()

        self.config = config

        # vocabularies
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']
        self.entity_type_stoi = vocabs['entity_type']
        self.entity_label_itos = {i:s for s, i in self.entity_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.entity_type_num = len(self.entity_type_stoi)
        self.char_num = len(vocabs['char2idx'])
        self.token_num = len(vocabs['word2idx'])

        # char embedding
        if config.use_char_emb:
            char_emb_dim = 25
            self.char_embeds = nn.Embedding(self.char_num, char_emb_dim)
            torch.nn.init.xavier_uniform_(self.char_embeds.weight)
            char_lstm_hidden = 25
            self.char_lstm = nn.LSTM(char_emb_dim, char_lstm_hidden, num_layers=1,
                                    batch_first=True, bidirectional=True)
        
        # word embedding
        if config.use_bert:
            # BERT encoder
            bert_config = config.bert_config
            bert_config.output_hidden_states = True
            self.bert_dim = bert_config.hidden_size
            self.bert_config = bert_config
            self.bert_dropout = nn.Dropout(p=config.bert_dropout)
            self.multi_piece = config.multi_piece_strategy
            self.load_bert(config.bert_model_name, config.bert_cache_dir)
            self.feature_dim = self.bert_dim+char_lstm_hidden*2 if config.use_char_emb else self.bert_dim
        else:
            word_emb_dim = 100
            if config.word2vec_path:
                pretrained_emb = torch.empty(self.token_num, word_emb_dim)
                nn.init.xavier_uniform_(pretrained_emb)
                cnt = 2
                for i, line in enumerate(open(config.word2vec_path, "r", encoding="utf-8")):
                    s = line.strip().split()
                    if len(s) == 101:
                        pretrained_emb[cnt].data = torch.from_numpy(np.array([float(f) for f in s[1:]]))
                        cnt += 1
                self.word_embeds = nn.Embedding.from_pretrained(pretrained_emb)
            else:
                self.word_embeds = nn.Embedding(self.token_num, word_emb_dim)
                torch.nn.init.xavier_uniform_(self.word_embeds.weight)
            lstm_input_dim = word_emb_dim+char_lstm_hidden*2 if config.use_char_emb else word_emb_dim
            lstm_hidden_dim = 128
            self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, num_layers=1,
                                batch_first=True, bidirectional=True)
            self.feature_dim = lstm_hidden_dim*2
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        self.entity_label_ffn = nn.Linear(self.feature_dim, self.entity_label_num,
                                            bias=linear_bias)
        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)
    
    def load_bert(self, name, cache_dir=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('Loading pre-trained BERT model {}'.format(name))
        if name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(name,
                                                  cache_dir=cache_dir,
                                                  config=self.bert_config)
        elif name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(name,
                                                  cache_dir=cache_dir,
                                                  config=self.bert_config)

    def load_pretrained_weight(self, pretrained_state):
        #ipdb.set_trace()
        for name, param in pretrained_state.items():
            if name in self.state_dict() and (not name.startswith('entity_label_ffn')) and (not name.startswith('entity_crf')):
                self.state_dict()[name].copy_(param)
            
            elif name in self.state_dict() and name.startswith('entity_label_ffn'):
                mean = torch.mean(param)
                std = torch.std(param)
                nn.init.normal_(self.state_dict()[name], mean=mean, std=std)

    def char_encode(self, char_idxs, char_lens, max_token_num):
        """Encode input char sequences
        :param char_idxs (list of 2D LongTensor): list of char indices
        :param char_lens (list of list): a list(batch) of list(how many tokens) of integer
        :param max_token_num (int): the maximum length of the sentence in the batch.
        """
        batch_size = len(char_idxs)
        batch_char_features = []
        for b_idx in range(len(char_idxs)):
            char_idx = char_idxs[b_idx] # #of_token x #max_len_char
            char_len = char_lens[b_idx] # len_char for each token
            char_embeds = self.char_embeds(char_idx)
            lstm_out, _  = self.char_lstm(char_embeds) #of_token x #max_len_char x lstm_dim
            # Get backward representation
            back_embed = lstm_out[:, 0, 25:]
            # Get forward representation
            length_vec = ((char_len.view(-1, 1).expand(char_idx.size(0), 25)).unsqueeze(1)) - 1 # -1 because it's length
            for_embed = ((lstm_out[:, :, :25]).gather(1, length_vec)).squeeze(1)
            pad_embed = back_embed.new_zeros(((max_token_num-len(char_idx)), 50), requires_grad=False)
            batch_char_features.append(
                torch.cat((
                    torch.cat((for_embed, back_embed), dim=-1), # #of_token x 2*lstm_dim
                    pad_embed
                ), dim=0)
            ) # #max_token_num x 2*lstm_dim
        return self.dropout(torch.stack(batch_char_features, dim=0))
    
    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets)
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def span_id(self, bert_outputs, batch, predict=False):
        loss = 0.0
        entities = None
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores_ = self.entity_crf.pad_logits(entity_label_scores)
        if predict:
            _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores_,
                                                                    batch.token_nums)
            entities = tag_paths_to_spans(entity_label_preds,
                                        batch.token_nums,
                                        self.entity_label_stoi,
                                        self.entity_type_stoi)
        else: 
            entity_label_loglik = self.entity_crf.loglik(entity_label_scores_,
                                                        batch.entity_label_idxs,
                                                        batch.token_nums)
            loss -= entity_label_loglik.mean()

        return loss, entities, bert_outputs
        #return loss, entities, F.softmax(entity_label_scores, dim=-1)
        #return loss, entities, entity_label_scores

    def forward(self, batch, logger=None, tag=None, step=None):
        # char encoding
        if self.config.use_char_emb:
            char_embeds = self.char_encode(batch.char_idxs, 
                                           batch.char_nums,
                                           max(batch.token_nums))
        
        # word encoding
        if self.config.use_bert:
            # encoding
            bert_outputs = self.encode(batch.piece_idxs,
                                       batch.attention_masks,
                                       batch.token_lens)
            # batch_size x max_len x bert_dim
            if self.config.use_char_emb:
                bert_outputs = torch.cat((bert_outputs, char_embeds), dim=-1)
        else:
            # lstm word encoding
            word_embeds = self.word_embeds(batch.word_idxs)
            if self.config.use_char_emb:
                word_embeds = torch.cat((word_embeds, char_embeds), dim=-1)
            word_embeds, _ = self.lstm(word_embeds)
            bert_outputs = self.dropout(word_embeds) # batch_size x max_len x 2*lstm_dim

        span_id_loss, _, _ = self.span_id(bert_outputs, batch, predict=False)
        loss = span_id_loss        
        if logger:
            logger.scalar_summary(tag+'/entity crf loss', span_id_loss, step)
        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # char encoding
            if self.config.use_char_emb:
                char_embeds = self.char_encode(batch.char_idxs, 
                                            batch.char_nums,
                                            max(batch.token_nums))
            
            # word encoding
            if self.config.use_bert:
                # encoding
                bert_outputs = self.encode(batch.piece_idxs,
                                        batch.attention_masks,
                                        batch.token_lens)
                # batch_size x max_len x bert_dim
                if self.config.use_char_emb:
                    bert_outputs = torch.cat((bert_outputs, char_embeds), dim=-1)
            else:
                # lstm word encoding
                word_embeds = self.word_embeds(batch.word_idxs)
                if self.config.use_char_emb:
                    word_embeds = torch.cat((word_embeds, char_embeds), dim=-1)
                word_embeds, _ = self.lstm(word_embeds)
                bert_outputs = self.dropout(word_embeds) # batch_size x max_len x 2*lstm_dim
                
            _, entities, output_prob = self.span_id(bert_outputs, batch, predict=True)

        self.train()
        return entities, output_prob


class Adapter(StructuralModel):
    def __init__(self,
                 config,
                 vocabs,
                 input_dim):
        super().__init__(config, vocabs)
        # self.adapter = nn.LSTM(input_size=input_dim, hidden_size=25, num_layers=1, batch_first=True, bidirectional=True)
        # self.adapter_linear = nn.Linear(2*25, self.entity_label_num)

        self.adapter = nn.LSTM(input_size=input_dim, hidden_size=self.feature_dim, num_layers=1, batch_first=True, bidirectional=True)
    
    def span_id_adapt(self, bert_outputs, adapter_out, batch, predict=False):
        loss = 0.0
        entities = None
        bert_outputs = bert_outputs + adapter_out
        entity_label_scores = self.entity_label_ffn(bert_outputs)
        #entity_label_scores = entity_label_scores + adapter_out
        entity_label_scores_ = self.entity_crf.pad_logits(entity_label_scores)
        if predict:
            _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores_,
                                                                    batch.token_nums)
            entities = tag_paths_to_spans(entity_label_preds,
                                        batch.token_nums,
                                        self.entity_label_stoi,
                                        self.entity_type_stoi)
        else: 
            entity_label_loglik = self.entity_crf.loglik(entity_label_scores_,
                                                        batch.entity_label_idxs,
                                                        batch.token_nums)
            loss -= entity_label_loglik.mean()

        return loss, entities, bert_outputs
        #return loss, entities, F.softmax(entity_label_scores, dim=-1)
        #return loss, entities, entity_label_scores

    def forward(self, batch, prior_prob=None, logger=None, tag=None, step=None):
        # char encoding
        if self.config.use_char_emb:
            char_embeds = self.char_encode(batch.char_idxs, 
                                           batch.char_nums,
                                           max(batch.token_nums))
        
        # word encoding
        if self.config.use_bert:
            # encoding
            bert_outputs = self.encode(batch.piece_idxs,
                                       batch.attention_masks,
                                       batch.token_lens)
            # batch_size x max_len x bert_dim
            if self.config.use_char_emb:
                bert_outputs = torch.cat((bert_outputs, char_embeds), dim=-1)
        else:
            # lstm word encoding
            word_embeds = self.word_embeds(batch.word_idxs)
            if self.config.use_char_emb:
                word_embeds = torch.cat((word_embeds, char_embeds), dim=-1)
            word_embeds, _ = self.lstm(word_embeds)
            bert_outputs = self.dropout(word_embeds) # batch_size x max_len x 2*lstm_dim

        if prior_prob is not None:
            # prior_prob is padded: batch x max_token_len x input_dim
            adapter_out, _ = self.adapter(prior_prob)
            #adapter_out = self.adapter_linear(adapter_out)
            adapter_out = adapter_out[:, :, :self.feature_dim] + adapter_out[:, :, self.feature_dim:]
            span_id_loss, _, _ = self.span_id_adapt(bert_outputs, adapter_out, batch, predict=False)
        else:
            span_id_loss, _, _ = self.span_id(bert_outputs, batch, predict=False)
        loss = span_id_loss        
        if logger:
            logger.scalar_summary(tag+'/entity crf loss', span_id_loss, step)
        return loss
    
    def predict(self, batch, prior_prob=None):
        self.eval()
        with torch.no_grad():
            # char encoding
            if self.config.use_char_emb:
                char_embeds = self.char_encode(batch.char_idxs, 
                                            batch.char_nums,
                                            max(batch.token_nums))
            
            # word encoding
            if self.config.use_bert:
                # encoding
                bert_outputs = self.encode(batch.piece_idxs,
                                        batch.attention_masks,
                                        batch.token_lens)
                # batch_size x max_len x bert_dim
                if self.config.use_char_emb:
                    bert_outputs = torch.cat((bert_outputs, char_embeds), dim=-1)
            else:
                # lstm word encoding
                word_embeds = self.word_embeds(batch.word_idxs)
                if self.config.use_char_emb:
                    word_embeds = torch.cat((word_embeds, char_embeds), dim=-1)
                word_embeds, _ = self.lstm(word_embeds)
                bert_outputs = self.dropout(word_embeds) # batch_size x max_len x 2*lstm_dim
            if prior_prob is not None:
                # prior_prob is padded: batch x max_token_len x input_dim
                adapter_out, _ = self.adapter(prior_prob)
                #adapter_out = self.adapter_linear(adapter_out)
                adapter_out = adapter_out[:, :, :self.feature_dim] + adapter_out[:, :, self.feature_dim:]
                _, entities, output_prob = self.span_id_adapt(bert_outputs, adapter_out, batch, predict=True)       
            else:
                _, entities, output_prob = self.span_id(bert_outputs, batch, predict=True)

        self.train()
        return entities, output_prob
