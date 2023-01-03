import json
import numpy as np
import os

def process_four_split_full_json(name_list,output_path):
    
    dict_type={
        "0": ["ORG", "LOC", "MISC"],
        "1": ["PER", "LOC", "MISC"],
        "2": ["PER", "ORG", "LOC"],
        "3": ["PER", "ORG", "MISC"]
    }

    name=["ORG_LOC_MISC","PER_LOC_MISC","PER_ORG_LOC","PER_ORG_MISC"]

    in_lines1 = []
    in_lines2 = []
    in_lines3 = []
    in_lines4 = []

    for i in name_list:
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines1.append(json.loads(line))
                in_lines2.append(json.loads(line))
                in_lines3.append(json.loads(line))
                in_lines4.append(json.loads(line))
        f_in.close()

    choose_file={
        "0": in_lines1,
        "1": in_lines2,
        "2": in_lines3,
        "3": in_lines4
    }
    
    for j in range(len(name)):
        extension = '.json'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = os.path.join(output_path, name[j]+extension)
        with open(file_path,'w',encoding='utf-8') as w:
            for k, line in enumerate(choose_file[str(j)]):
                answer = line
                answer_=[]
                for m in answer['entity_mentions']:
                    if(m["entity_type"]  in dict_type[str(j)]):
                        answer_.append(m)
                answer['entity_mentions']=answer_
                w.write(json.dumps(answer) + "\n") 
      
def process_six_split_full_json(name_list,output_path):
    
    dict_type={
        "0": ["LOC", "MISC"],
        "1": ["PER", "ORG"],
        "2": ["LOC", "ORG"],
        "3": ["PER", "MISC"],
        "4": ["LOC", "PER"],
        "5": ["ORG", "MISC"],   
    }

    name=["LOC_MISC","PER_ORG","LOC_ORG","PER_MISC","LOC_PER","ORG_MISC"]

    in_lines1 = []
    in_lines2 = []
    in_lines3 = []
    in_lines4 = []
    in_lines5 = []
    in_lines6 = []

    for i in name_list:
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines1.append(json.loads(line))
                in_lines2.append(json.loads(line))
                in_lines3.append(json.loads(line))
                in_lines4.append(json.loads(line))
                in_lines5.append(json.loads(line))
                in_lines6.append(json.loads(line))
        f_in.close()
    
    choose_file={
        "0": in_lines1,
        "1": in_lines2,
        "2": in_lines3,
        "3": in_lines4,
        "4": in_lines5,
        "5": in_lines6
    }
    for j in range(len(name)):
        extension = '.json'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = os.path.join(output_path, name[j]+extension)
        with open(file_path,'w',encoding='utf-8') as w:
            for k, line in enumerate(choose_file[str(j)]):
                answer = line
                answer_=[]
                for m in answer['entity_mentions']:
                    if(m["entity_type"]  in dict_type[str(j)]):
                        answer_.append(m)
                answer['entity_mentions']=answer_
                w.write(json.dumps(answer) + "\n") 

def process_three_split_partial_json(name_list,output_path):

    K_FOLD = 3

    dict_type={
        "0": ["PER", "ORG", "LOC"],
        "1": ["ORG", "LOC", "MISC"],
        "2": ["MISC", "ORG", "LOC"],
    }

    in_lines1 = []
    in_lines2 = []
    in_lines3 = []

    for i in name_list:
        type=i.split("/")[2]
   
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines1.append(json.loads(line))
                in_lines2.append(json.loads(line))
                in_lines3.append(json.loads(line))
        f_in.close()

        choose_file={
            "0": in_lines1,
            "1": in_lines2,
            "2": in_lines3,
        }

        for j in range(0, K_FOLD):
            extension = '.json'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_path = os.path.join(output_path, str(type)+"+"+str(j)+extension)

            choose=choose_file[str(j)]

            if j == K_FOLD-1:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(choose[int(round(K_FOLD - 1 / K_FOLD * len(choose))): ]):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
                w.close()

            else:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(choose[int(round(j / K_FOLD * len(choose))): int(round((j + 1)/ K_FOLD * len(choose)))]):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
                w.close()               

def process_four_split_partial_json(name_list,output_path):

    K_FOLD = 4

    dict_type={
        "0": ["ORG", "LOC", "MISC"],
        "1": ["PER", "LOC", "MISC"],
        "2": ["PER", "ORG", "LOC"],
        "3": ["PER", "ORG", "MISC"]
    }

    in_lines1 = []
    in_lines2 = []
    in_lines3 = []
    in_lines4 = []

    for i in name_list:
        type=i.split("/")[2]
   
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines1.append(json.loads(line))
                in_lines2.append(json.loads(line))
                in_lines3.append(json.loads(line))
        f_in.close()

        choose_file={
            "0": in_lines1,
            "1": in_lines2,
            "2": in_lines3,
            "3": in_lines4,
        }

        for j in range(0, K_FOLD):
            extension = '.json'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_path = os.path.join(output_path, str(type)+"+"+str(j)+extension)

            choose=choose_file[str(j)]

            if j == K_FOLD-1:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(choose[int(round(K_FOLD - 1 / K_FOLD * len(choose))): ]):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
                w.close()

            else:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(choose[int(round(j / K_FOLD * len(choose))): int(round((j + 1)/ K_FOLD * len(choose)))]):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
                w.close()               

def process_six_split_partial_json(name_list,output_path):

    K_FOLD = 6

    dict_type={
        "0": ["LOC", "MISC"],
        "1": ["PER", "ORG"],
        "2": ["LOC", "ORG"],
        "3": ["PER", "MISC"],
        "4": ["LOC", "PER"],
        "5": ["ORG", "MISC"],   
    }

    in_lines1 = []
    in_lines2 = []
    in_lines3 = []
    in_lines4 = []
    in_lines5 = []
    in_lines6 = []

    for i in name_list:
        type=i.split("/")[2]
   
        with open(i + ".json",'r',encoding='utf-8') as f_in:
            for line in f_in:
                in_lines1.append(json.loads(line))
                in_lines2.append(json.loads(line))
                in_lines3.append(json.loads(line))
        f_in.close()

        choose_file={
            "0": in_lines1,
            "1": in_lines2,
            "2": in_lines3,
            "3": in_lines4,
            "4": in_lines5,
            "5": in_lines6
        }

        for j in range(0, K_FOLD):
            extension = '.json'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_path = os.path.join(output_path, str(type)+"+"+str(j)+extension)

            choose=choose_file[str(j)]

            if j == K_FOLD-1:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(choose[int(round(K_FOLD - 1 / K_FOLD * len(choose))): ]):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
                w.close()

            else:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(choose[int(round(j / K_FOLD * len(choose))): int(round((j + 1)/ K_FOLD * len(choose)))]):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
                w.close()               

def process_progressive_learn_json(name_list,ratio,option):
    K_FOLD = 2

    state1_label=["PER","LOC", "MISC","ORG"]
    state0_label=["PER","LOC", "MISC","ORG"]
    state0_label.remove(option)  #if you need just split and not drop labels, you should comment out this line
    # print(state1_label)
    # print(state0_label)

    dict_type={"0":state0_label, "1":state1_label}

    outputs_path="processed/" 
    splits=str(int(ratio*100))+str(int(100-ratio*100))
    
    type_maps={"LOC":"location",
                "ORG":"organization",
                "PER":"person",
                "MISC":"miscellaneous"
    }

    types=type_maps[option]

    output_path=os.path.join(outputs_path,"progressive_learn_"+splits+types+"_json")
    print(output_path)

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
            extension = '.json'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            file_path = os.path.join(output_path, str(type)+"+"+str(j)+extension)

            if j ==  K_FOLD - 1:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(dataset_1):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
            else:
                with open(file_path,'w',encoding='utf-8') as w:
                    for k, line in enumerate(dataset_0):
                        answer = line
                        answer_=[]
                        for m in answer['entity_mentions']:
                            if(m["entity_type"]  in dict_type[str(j)]):
                                answer_.append(m)
                        answer['entity_mentions']=answer_
                        w.write(json.dumps(answer) + "\n") 
