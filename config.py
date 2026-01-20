ROOT_PATH = '[YOUR PATH OF PROJECT] (e.g., ~/NextMem)'
Qwen_3_8B_model_path = '[MODEL PATH]'
DeepSeekOCR_path = '[MODEL PATH]'
Mistral_7B_model_path = '[MODEL PATH]'
Llama_3_8B_model_path = '[MODEL PATH]'
Llama_3_1_8B_model_path = '[MODEL PATH]'

TASK_1_DATA_PATH = {
    'squad': f'{ROOT_PATH}/datasets/recs_squad_test.json',
    'hotpot': f'{ROOT_PATH}/datasets/recs_hotpot_test.json',
    'race': f'{ROOT_PATH}/datasets/recs_race_test.json',
    'longmemeval': f'{ROOT_PATH}/datasets/recs_longmemeval_test.json',
    'locomo': f'{ROOT_PATH}/datasets/recs_locomo_test.json',
}

TASK_2_DATA_PATH = {
    'squad': f'{ROOT_PATH}/datasets/stm_squad_test.json',
    'hotpot': f'{ROOT_PATH}/datasets/stm_hotpot_test.json',
    'race': f'{ROOT_PATH}/datasets/stm_race_test.json',
    'longmemeval': f'{ROOT_PATH}/datasets/stm_longmemeval_test.json',
    'locomo': f'{ROOT_PATH}/datasets/stm_locomo_test.json'
}

TASK_3_DATA_PATH = {
    'hotpot': f'{ROOT_PATH}/datasets/hotpotqa_ltm_test.json',
    'longmemeval': f'{ROOT_PATH}/datasets/longmemeval_s_ltm_test.json',
    'locomo': f'{ROOT_PATH}/datasets/locomo_ltm_test.json'
}


DEFAULT_CONFIG = {
    'NextMemQwen': {
        'model_name': 'NextMemQwen',
        'model_path': Qwen_3_8B_model_path,
        'stage_1_checkpoint_path': f'{ROOT_PATH}/checkpoints/NextMemQwen-L15/stage_1_checkpoint-3092',
        'stage_2_checkpoint_path': f'{ROOT_PATH}/checkpoints/NextMemQwen-L15/stage_2_checkpoint-15',
        'device': 'cuda',
        'max_latent_length': 15,
        'max_encode_length': 1024,
        'max_length': 1024
    },
    'DeepSeekOCR': {
        'model_name': 'DeepSeekOCR',
        'model_path': DeepSeekOCR_path,
        'cache_path': f'{ROOT_PATH}/baselines/checkpoints/DeepSeekOCR',
        'img_size': 240,
    },
    'NextMemQwenSparse': {
        'model_name': 'NextMemQwenSparse',
        'model_path': Qwen_3_8B_model_path,
        'stage_1_checkpoint_path': f'{ROOT_PATH}/checkpoints/NextMemQwen-L15/stage_1_checkpoint-3092',
        'stage_2_checkpoint_path': f'{ROOT_PATH}/checkpoints/NextMemQwen-L15/stage_2_checkpoint-15',
        'device': 'cuda',
        'max_latent_length': 15,
        'max_encode_length': 1024,
        'max_length': 1024
    },
    'ICAE': {
        'model_args': {
            'model_name_or_path': Mistral_7B_model_path,
            'lora_r': 512,
            'lora_dropout': 0.05,
            'train': False
            },
        'training_args': {
            'optim': 'adamw_torch',
            'model_max_length': 1024,
            'max_new_length': 256,
            'fixed_mem_size': 128,
            'mean_compression_rate': 4,
            'min_tokens_for_lm': 64,
            'leave_tokens_for_lm': 8,
            'lm_ratio': 0.0,
            'add_special_token_for_lm': False,
            'restore_from': "",
            'bf16': True,
            'output_path': f'{ROOT_PATH}/checkpoints/ICAE/mistral_7b_ft_icae.safetensors',
        },
    },
    'DyPRAG': {
        'model_name': 'DyPRAG',
        'model_args': {
            'model_path': Llama_3_8B_model_path,
            'lora_rank': 2,
            'lora_alpha': 32,
            'projector_path': f'{ROOT_PATH}/checkpoints/DyPRAG/llama3-8b-p32-1ep-main-2400sample.pt',
            'max_length': 1024,
            'projector_p': 32,
            'generation_config': {
                'num_beams': 1,
                'do_sample': False,
                'max_new_tokens': 1024,
                'return_dict_in_generate': True,
            }
        },
    }
}