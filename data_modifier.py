import json
import os


def load_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

data_type = 'train'

path = f'datasets/selected/{data_type}'

subjects = os.listdir(path)



expert_1_result = []
expert_2_result = []

for s in subjects :
    files = os.listdir(os.path.join(path, s))
    files_data = [load_json_file(os.path.join(path, s, file)) for file in files if file.endswith('.json')]

    for f in files_data :
        for k in f['rubric']['analytic'].keys() :
            


            result_1 = {
                "id" : f['essay_question']['id'],
                "question" : f['essay_question']['prompt'],
                "response" : f['essay_answer']['text'],
                "score" : f['score']['personal']['analytic'][k]['score'][0],
                "rubric" : f['rubric']['analytic'][k]
            }

            expert_1_result.append(result_1)

            result_2 = {
                "id" : f['essay_question']['id'],
                "question" : f['essay_question']['prompt'],
                "response" : f['essay_answer']['text'],
                "score" : f['score']['personal']['analytic'][k]['score'][1],
                "rubric" : f['rubric']['analytic'][k]
            }

            expert_2_result.append(result_2)


for i in range(2) :
    result_path = os.path.join(f"{data_type}_data", f"{data_type}_essay_qas_{data_type}_expert_{i}.json")

    with open(result_path, "w", encoding= "utf-8") as f :
        json.dump(expert_1_result if i == 0 else expert_2_result, f, ensure_ascii = False, indent = 4)
