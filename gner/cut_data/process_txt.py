import json
import numpy as np
import os

#task1
def process_progressive_learn_not_drop_txt(name_list,output_path,ratio):
    K_FOLD = 2

    name=["Fold0/","Fold1/"]

    splits=str(int(ratio*100))+str(int(100-ratio*100))

    for i in name_list:
        type=i.split("/")[2]
        in_lines = []
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines.append(json.loads(line))
        f_in.close()

        dataset_0=in_lines[:round(ratio*len(in_lines))] 
        dataset_1=in_lines[round(ratio*len(in_lines)):]

        for j in range(0, K_FOLD):
            extension = '.txt'
            outputs_path = os.path.join(output_path, name[j])
            if not os.path.exists(outputs_path):
                os.makedirs(outputs_path)
            file_path = os.path.join(outputs_path, type+extension)

            if j ==  K_FOLD - 1:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(dataset_1):
                        answer = line
                        sentences= answer['sentence'].split()
                        text_label = np.empty(len(sentences), dtype=object)
                        for l in range(len(sentences)):
                            text_label[l]=str(sentences[l] + ' ' + 'O'+ '\n')
                        for m in answer['entity_mentions']:
                            start1=int(m["start"])
                            end1=int(m["end"])
                            type1=m["entity_type"] 
                            text_label[start1]=str(sentences[start1] + ' ' + 'B-'+ str(type1)+ '\n')
                            for o in range(start1+1,end1):
                                text_label[o] = str(sentences[o] + ' ' + 'I-'+ str(type1)+ '\n')
                        for q in range(len(sentences)):
                            w.write(text_label[q])
                        w.write('\n')
            else:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(dataset_0):
                        answer = line
                        sentences= answer['sentence'].split()
                        text_label = np.empty(len(sentences), dtype=object)
                        for l in range(len(sentences)):
                            text_label[l]=str(sentences[l] + ' ' + 'O'+ '\n')
                        for m in answer['entity_mentions']:
                            start1=int(m["start"])
                            end1=int(m["end"])
                            type1=m["entity_type"] 
                            text_label[start1]=str(sentences[start1] + ' ' + 'B-'+ str(type1)+ '\n')
                            for o in range(start1+1,end1):
                                text_label[o] = str(sentences[o] + ' ' + 'I-'+ str(type1)+ '\n')
                        for q in range(len(sentences)):
                            w.write(text_label[q])
                        w.write('\n')
#task2
def process_three_split_partial_txt(name_list,output_path):
    
    K_FOLD = 3

    dict_type={
        "0": ["PER", "ORG", "LOC"],
        "1": ["ORG", "LOC", "MISC"],
        "2": ["MISC", "ORG", "LOC"],
    }

    name=["Fold0/","Fold1/","Fold2/"]

    for i in name_list:
        type=i.split("/")[2]

        in_lines = []
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines.append(json.loads(line))
        f_in.close()

        for j in range(0, K_FOLD):
            extension = '.txt'
            outputs_path = os.path.join(output_path, name[j])
            if not os.path.exists(outputs_path):
                os.makedirs(outputs_path)
            file_path = os.path.join(outputs_path, type+extension)

            if j ==  K_FOLD - 1:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(in_lines[int(round(K_FOLD - 1 / K_FOLD * len(in_lines))): ]):
                        answer = line
                        sentences= answer['sentence'].split()
                        text_label = np.empty(len(sentences), dtype=object)
                        for l in range(len(sentences)):
                            text_label[l]=str(sentences[l] + ' ' + 'O'+ '\n')
                        for m in answer['entity_mentions']:
                            start1=int(m["start"])
                            end1=int(m["end"])
                            type1=m["entity_type"] 
                            if(type1 in dict_type[str(j)]):
                                text_label[start1]=str(sentences[start1] + ' ' + 'B-'+ str(type1)+ '\n')
                                for o in range(start1+1,end1):
                                    text_label[o] = str(sentences[o] + ' ' + 'I-'+ str(type1)+ '\n')
                        for q in range(len(sentences)):
                            w.write(text_label[q])
                        w.write('\n')

            else:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(in_lines[int(round(j / K_FOLD * len(in_lines))): int(round((j + 1)/ K_FOLD * len(in_lines)))]):
                        answer = line
                        sentences= answer['sentence'].split()
                        text_label = np.empty(len(sentences), dtype=object)
                        for l in range(len(sentences)):
                            text_label[l]=str(sentences[l] + ' ' + 'O'+ '\n')
                        for m in answer['entity_mentions']:
                            start1=int(m["start"])
                            end1=int(m["end"])
                            type1=m["entity_type"] 
                            if(type1 in dict_type[str(j)]):
                                text_label[start1]=str(sentences[start1] + ' ' + 'B-'+ str(type1)+ '\n')
                                for o in range(start1+1,end1):
                                    text_label[o] = str(sentences[o] + ' ' + 'I-'+ str(type1)+ '\n')
                        for q in range(len(sentences)):
                            w.write(text_label[q])
                        w.write('\n')
#task3

