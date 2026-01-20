from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, gc
import torch.nn as nn
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
from collections import defaultdict

def get_attributes(x: nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.encoder.layer')
        should return the same as model.transformer.encoder.layer
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x

def delta_inject(model, adapter_weights):
    """
    Injects delta weights into the model's layers.
    
    Args:
        model: The model to inject deltas into.
        adapter_weights: A dictionary containing the delta weights.
    """
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        lora_A = adapter_weights[module + ".lora_A.weight"]
        lora_B = adapter_weights[module + ".lora_B.weight"]
        # Calculate delta
        delta = lora_B @ lora_A
        # Set the delta in the module
        setattr(m, "delta", delta.to(torch.float32))  
                
def delta_remove(model, adapter_weights):
    """
    Removes delta weights from the model's layers.
    
    Args:
        model: The model to remove deltas from.
        adapter_weights: A dictionary containing the delta weights.
    """
    modules = set(".".join(k.split(".")[:-2]) for k in adapter_weights.keys())
    for module in modules:
        m = get_attributes(model, module)
        delattr(m, "delta") 


class ParameterTranslator(nn.Module):
    def __init__(self, module_list: list[str], layer_idx: list[int], input_dim: int, output_dim: int, lora_rank: int, hidden_dim: int=32):
        super().__init__()
        self.module_list = module_list
        self.layer_idx = layer_idx
        self.projector = nn.ModuleDict()
        for module_name in self.module_list:
            for layer_idx in self.layer_idx:
                self.projector[f"{module_name}_{layer_idx}"] = Projector(module_name, layer_idx, input_dim, output_dim,  lora_rank, hidden_dim)
    
    def forward(self, x):
        ret = defaultdict(list)
        for module_name in self.module_list:
            for layer_idx in self.layer_idx:
                lora_A, lora_B = self.projector[f"{module_name}_{layer_idx}"](x)
                ret[f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_A.weight"] = lora_A
                ret[f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}.lora_B.weight"] = lora_B
        return ret
    
class Projector(nn.Module):
    def __init__(self, module_name, layer_idx, input_dim, output_dim, lora_rank, hidden_dim=8):
        super().__init__()
        self.module_name = module_name
        self.layer_idx = layer_idx
        self.projector = ProjectorLoRA(module_name, layer_idx, input_dim, lora_rank, output_dim, hidden_dim)
    
    def forward(self, x):
        idxs_tensor = torch.tensor(self.layer_idx, device=x.device, dtype=torch.float32).view(-1, 1)
        network_input = torch.cat([x, idxs_tensor], dim=1)
        self.lora_A = self.projector.A_hypernet(network_input)
        self.lora_B = self.projector.B_hypernet(network_input)
        return self.lora_A, self.lora_B            
           

class ProjectorLoRA(nn.Module):
    def __init__(self, module_name, layer_idx, input_dim, lora_rank, output_dim, hidden_dim=16):
        super().__init__()
        self.module_name = module_name
        self.layer_idx = layer_idx
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        #Initialize the weight of the LoRA A projector
        self.pre_A_linear = nn.Linear(input_dim + 1, hidden_dim, bias=False, dtype=torch.float32)
        self.pre_A_linear.weight = self.init_layer(self.pre_A_linear)
        if self.module_name == "down_proj":
            self.post_A_linear = nn.Linear(hidden_dim, output_dim * lora_rank, bias=False, dtype=torch.float32)
        else:
            self.post_A_linear = nn.Linear(hidden_dim, input_dim * lora_rank, bias=False, dtype=torch.float32)
        self.post_A_linear.weight = self.init_layer(self.post_A_linear)
        if self.module_name == "down_proj":
            self.A_hypernet = MLPHypernet(self.pre_A_linear, self.post_A_linear, lora_rank, output_dim)
        else:
            self.A_hypernet = MLPHypernet(self.pre_A_linear, self.post_A_linear, lora_rank, input_dim)
        
        #Initialize the weight of the LoRA B projector
        self.pre_B_linear = nn.Linear(input_dim + 1, hidden_dim, bias=False, dtype=torch.float32)
        self.pre_B_linear.weight = self.init_layer(self.pre_B_linear)
        if self.module_name == "down_proj": 
            self.post_B_linear = nn.Linear(hidden_dim, input_dim * lora_rank, bias=False, dtype=torch.float32)
        else:
            self.post_B_linear = nn.Linear(hidden_dim, output_dim * lora_rank, bias=False, dtype=torch.float32)
        self.post_B_linear.weight = self.init_layer(self.post_B_linear)
        if self.module_name == "down_proj":
            self.B_hypernet = MLPHypernet(self.pre_B_linear, self.post_B_linear, input_dim, lora_rank)
        else:
            self.B_hypernet = MLPHypernet(self.pre_B_linear, self.post_B_linear, output_dim, lora_rank)
        
    def init_layer(self, layer):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        return weight
    

class MLPHypernet(nn.Module):
    def __init__(self, linear1, linear2, input_dim, output_dim):
        super().__init__()
        self.linear1 = linear1
        self.linear2 = linear2
        self.input_dim = input_dim  
        self.output_dim = output_dim  
    def forward(self, features):
        output = self.linear2(F.relu(self.linear1(features))).reshape(self.input_dim, self.output_dim)
        return output

class DyPRAG():
    def __init__(self, config):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_args']['model_path'], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_args']['model_path'],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto", 
            trust_remote_code=True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config['model_args']['generation_config']['pad_token_id'] = self.tokenizer.pad_token_id

        self.model = get_peft_model(self.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=self.config['model_args']['lora_rank'],
            lora_alpha=self.config['model_args']['lora_alpha'],
            lora_dropout=0,
        ))

        self.projector = ParameterTranslator(
            ["down_proj", "up_proj", "gate_proj"],
            list(range(self.model.config.num_hidden_layers)),
            self.model.config.hidden_size,
            self.model.config.intermediate_size,
            self.config['model_args']['lora_rank'],
            self.config['model_args']['projector_p']
        ).to(self.model.device)

        self.projector.load_state_dict(
            torch.load(self.config['model_args']['projector_path'],
                       map_location=self.model.device)['model_state_dict'])
        self.projector.eval()

    def encode(self, text):
        all_deltas = []

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config['model_args']['max_length']
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model(tokens.input_ids, output_hidden_states=True)
            input_embeds = output.hidden_states[-1][:,-1,:]
            outputs = self.projector(input_embeds)
            all_deltas.append(outputs)
        merged_deltas = {}
        for key in all_deltas[0].keys():
            merged_deltas[key] = torch.stack([delta[key] for delta in all_deltas]).mean(dim=0)
        
        return merged_deltas, all_deltas

    def decode(self, representation, prompt='Repeat the input.'):
        merged_deltas, all_deltas = representation

        delta_inject(self.model, merged_deltas)

        # --- Inference Process ---
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            add_generation_prompt=True)
        input_len = len(input_ids)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids, 
                attention_mask = torch.ones(input_ids.shape).to(self.model.device),
                **self.config['model_args']['generation_config'])
        output = output.sequences[0][input_len:]
        res = self.tokenizer.decode(output, skip_special_tokens=True)
        # --- End ---

        delta_remove(self.model, merged_deltas)
        del all_deltas, merged_deltas
        torch.cuda.empty_cache()
        gc.collect()

        return res
    
    def inference(self, representation, question):
        return self.decode(representation, prompt = question)

    # def index(self, text):
    #     lora_a_list, lora_b_list = [], []
    #     merged_deltas, all_deltas = self.encode(text)
    #     for k,v in merged_deltas.items():
    #         print(k, v.shape)
    #         if 'lora_A' in k:
    #             lora_a_list.append(v)
    #         elif 'lora_B' in k:
    #             lora_b_list.append(v)
    #     lora_a_list = torch.stack(lora_a_list, dim=0)
    #     lora_b_list = torch.stack(lora_b_list, dim=0)
    #     print(lora_a_list.shape, lora_b_list.shape)
    #     raise

    #     del all_deltas, merged_deltas
    #     torch.cuda.empty_cache()
    #     gc.collect()