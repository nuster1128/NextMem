import importlib, json, sys, random, argparse, os
sys.path.append('../..')
from config import DEFAULT_CONFIG, TASK_1_DATA_PATH, ROOT_PATH

import numpy as np
import torch

random_seed = 1128
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

dataset_list = ['squad', 'hotpot', 'race', 'locomo', 'longmemeval']

def save_jsonl_add(path, data_dict):
    with open(path, 'a', encoding='utf-8') as f:
        json_line = json.dumps(data_dict, ensure_ascii=False)
        f.write(json_line + '\n')

def load_dataset(dataset_name):
    data_path = TASK_1_DATA_PATH[dataset_name]
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data

def single_evaluation(model, dataset, save_path):
    print(f'----- Evaluation {model_name} on {dataset_name} -----')

    for index, ref in enumerate(dataset):
        representation = model.encode(ref)
        reconstruction = model.decode(representation)

        save_jsonl_add(save_path, {
            'index': index,
            'reference': ref,
            'reconstruction': reconstruction,
            'word_num': len(ref.split()),
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