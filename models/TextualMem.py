from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen3Model():
    def __init__(self, config):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'],
            dtype=torch.bfloat16,
            device_map='cuda'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_path']
        )
    
    def inference(self, prompt):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response

prompt_template = """Please answer the following question based on the reference.
Reference:
{reference}

Question:
{question}

Answer:"""

class TextualMem():
    def __init__(self, config):
        self.model = Qwen3Model({'model_path': config['model_path']})

    def encode(self, text):
        return text

    def decode(self, representation):
        raise

    def inference(self, representation, question):
        return self.model.inference(prompt_template.format(reference=representation, question=question))