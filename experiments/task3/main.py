import importlib, json, sys, random, argparse, os
sys.path.append('../..')
from config import DEFAULT_CONFIG, TASK_3_DATA_PATH, ROOT_PATH

import numpy as np
import torch

random_seed = 1128
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

sample_gap = {
    'longmemeval': 1,
    'hotpot': 7 * 4,
    'locomo': 1
}

dataset_list = ['hotpot', 'locomo', 'longmemeval']

def save_jsonl_add(path, data_dict):
    with open(path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_line + '\n')

def load_dataset(dataset_name):
    data_path = TASK_3_DATA_PATH[dataset_name]
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data

def single_evaluation(model, dataset, save_path):
    print(f'----- Evaluation {model_name} on {dataset_name} -----')

    for index, item in enumerate(dataset[::sample_gap[dataset_name]]):

        question, references, hit_list = item['question'], item['references'], item['hit_list']

        vectorbase = []
        for reference in references:
            ref_index = model.index(reference)
            vectorbase.append(ref_index)

        vectorbase = torch.stack(vectorbase, dim=0)

        query_index = model.index(question)
        # calculate cosine similarity and find top-5
        sim_scores = torch.cosine_similarity(query_index, vectorbase, dim=-1)
        top5_indices = torch.topk(sim_scores, k=min(5, len(references)), dim=-1).indices.tolist()

        save_jsonl_add(save_path, {
            'index': index,
            'question': question,
            'hit_list': hit_list,
            'prediction': top5_indices,
            'ref_num': len(references)
        })

        print(f'[{index}] has finished.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name')
    args = parser.parse_args()

    model_name = args.model
    model_cls = getattr(importlib.import_module(f'models.{model_name}'), f'{model_name}')
    model = model_cls(DEFAULT_CONFIG[model_name])

    if not os.path.exists(f'./results'):
        os.makedirs(f'./results')

    for dataset_name in dataset_list:    
        dataset = load_dataset(dataset_name)
        save_path = f'./results/{model_name}_{dataset_name}.jsonl'
        single_evaluation(model, dataset, save_path)