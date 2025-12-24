import json
import os


def load_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

data_type = 'val'
data_name = "mid_1_society"

path = f'datasets/selected/{data_type}'

files_path = os.path.join(path, data_name)
files = os.listdir(files_path)

files_data = [load_json_file(os.path.join(files_path, file)) for file in files if file.endswith('.json')]

results = []

for f in files_data :

    result = {
        "id" : f['essay_question']['id'],
        "question" : f['essay_question']['prompt'],
        "response" : f['essay_answer']['text'],
        "score" : f['score']['personal']['holistic']['score']
    }

    results.append(result)

result_path = os.path.join(f"{data_type}_data", f"{data_name}_essay_qas_eval.json")

with open(result_path, "w", encoding= "utf-8") as f :
    json.dump(results, f, ensure_ascii = False, indent = 4)
