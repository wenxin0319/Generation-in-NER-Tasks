import sys
import ipdb
import json

INPUT_STYLE_SET = ['task_def', 'template']

class base_template():
    task_name = 'BASE_TEMPLATE'
    dataset_name = 'BASE_TEMPLATE'
    
    def __init__(self, input_style, tokens, class_options, ground_truth=[]):
        self.input_style = input_style
        self.passage = ' '.join(tokens)
        self.tokens = tokens
        self.class_options = class_options
        self.ground_truth = ground_truth
        
    @classmethod
    def get_keywords(self):
        '''
        This is deprecated
        '''
        pass

    def task_def_prompt(self, query_subject):
        pass

    def output_template(self, query_subject):
        pass

    def generate_pair(self, query_subject):
        """
        Generate model input sentence and output sentence pair
        """
        input_str = self.generate_input_str(query_subject)
        output_str, gold_sample = self.generate_output_str(query_subject)
        return (input_str, output_str, self.ground_truth, gold_sample, self.tokens)

    def generate_input_str(self, query_subject):
        return None

    def generate_output_str(self, query_subject):
        return (None, False)

    def decode(self, pred_sent):
        pass

class seqtag_template(base_template):
    task_name = 'SEQTAG_TEMPLATE'
    dataset_name = 'SEQTAG_TEMPLATE'
    
    def __init__(self, input_style, tokens, class_options, ground_truth=[]):
        super().__init__(input_style, tokens, class_options, ground_truth)
    
    def generate_input_str(self, query_subject):
        input_str = ''
        for i_style in INPUT_STYLE_SET:
            if i_style in self.input_style:
                if i_style == 'task_def':
                    input_str += '{} [SEP] '.format(self.task_def_prompt(query_subject))

                if i_style == 'template':
                    input_str += '{} [SEP] '.format(self.output_template(query_subject))
        input_str += self.passage
        return input_str

class NERCoNLL_template(seqtag_template):
    class2name = {
        'MISC': 'miscellaneous',
        'ORG': 'organization', 
        'PER': 'person',
        'LOC': 'location'
    }
    name2class = {v:k for k,v in class2name.items()}
    
    def __init__(self, input_style, tokens, class_options, ground_truth=[]):
        super().__init__(input_style, tokens, class_options, ground_truth)
        
    def task_def_prompt(self, query_subject):
        return "Identify and categorize entities in text."

    def output_template(self, query_subject):
        sentences = []
        for c in self.class_options:
            sentences.append(self.class2name[c]+" : [None]")
        return ' [CLS_SEP] '.join(sentences)

    def generate_output_str(self, query_subject):
        sentences = []
        gold_sample = False
        collects = {}
        for ent in self.ground_truth:
            gold_sample = True
            if ent['entity_type'] not in collects.keys():
                collects[ent['entity_type']] = []
            collects[ent['entity_type']].append(' '.join(self.tokens[ent['start']:ent['end']]))
        for c in self.class_options:
            if c in collects.keys():
                sentences.append(self.class2name[c]+" : {}".format(' [and] '.join(collects[c])))
            else:
                sentences.append(self.class2name[c]+" : [None]")
        return ' [CLS_SEP] '.join(sentences), gold_sample

    def decode(self, pred_sent):
        pred_sent = pred_sent.split(' [SEP] ', 1)[-1]
        class_split = pred_sent.split(' [CLS_SEP] ')
        output = []
        for sp in class_split:
            try:
                class_name = self.name2class[sp.split(' : ',1)[0]]
                objects = (sp.split(' : ',1)[1]).split(' [and] ')
                for obj in objects:
                    if obj != '[None]':
                        output.append((obj, class_name))
            except:
                pass
        return output
