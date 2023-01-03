import json
import os

def process_different_type(name_list,output_path):

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
                    for m in answer['entity_mentions']:
                        if m["entity_type"] == 'O':
                            pass
                        elif m["entity_type"] == 'PER':
                            m["entity_type"] = "person"
                        elif m["entity_type"] == 'LOC':
                            m["entity_type"] = "location"
                        elif m["entity_type"] == 'ORG':
                            m["entity_type"] = "organization"
                        elif m["entity_type"] == 'MISC':
                            m["entity_type"] = "miscellaneous"
                        else:
                            print("error")

                    w.write(json.dumps(answer) + "\n")              