import argparse, torch, math
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from peft import get_peft_model, LoraConfig
from typing import Optional

class ICAE():
    def __init__(self, config):
        self.config = config

        self.model_args = argparse.Namespace(**config['model_args'])
        self.training_args = argparse.Namespace(**config['training_args'])

        self.model = ICAEModel(self.model_args, self.training_args, LoraConfig(
            r=512,
            lora_alpha=32,
            lora_dropout=self.model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        ))

        print(f"Loading trained checkpoint from {self.training_args.output_path}")
        state_dict = load_file(self.training_args.output_path)
        self.model.load_state_dict(state_dict, strict=False) # only load lora and memory token embeddings

        self.model = self.model.cuda()

    def encode(self, text):
        tokenized_input = self.model.tokenizer(text, truncation=True, max_length=self.training_args.model_max_length, padding=False, return_attention_mask=False)
        # Generate compressed outputs
        input_ids = torch.LongTensor([tokenized_input['input_ids']]).cuda()
        memory_slots = self.model._compress(input_ids)

        return memory_slots      

    def decode(self, representation, prompt='Repeat the input.'):

        memory_slots = representation
        tokenized_prompt = self.model.tokenizer(prompt, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)

        prompt_left_ids =  torch.LongTensor([[1, 733, 16289, 28793]]).cuda()
        prompt_right_ids = [self.model.ft_token_id] + tokenized_prompt['input_ids'] + [733, 28748, 16289, 28793]
        prompt_right_ids = torch.LongTensor([prompt_right_ids]).cuda()

        prompt_left_embs = self.model.tokens_to_embeddings(prompt_left_ids)
        prompt_right_embs = self.model.tokens_to_embeddings(prompt_right_ids)
        memory_slots = memory_slots.to(prompt_right_embs)
                    
        # Concatenate and clone input embeddings
        decoder_input_embeddings = torch.cat((prompt_left_embs, memory_slots.unsqueeze(0), prompt_right_embs), dim=1)
        representation = decoder_input_embeddings.clone()
        # print(output.shape)
    
        generate_text = []
        past_key_values = None

        # Generate text output
        for i in range(self.training_args.max_new_length):
            # print('--- DEBUG00 ---')
            with self.model.icae.disable_adapter():   # no independent decoder; use self.icae
                out = self.model.icae(inputs_embeds=representation, past_key_values=past_key_values, use_cache=True)
            # out = decoder(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
            logit = out.logits[:, -1, :self.model.vocab_size-1]
            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logit, dim=-1)
            # print(next_token_id)
            
            if next_token_id.item() == 2:   # eos
                break

            representation = self.model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).cuda()
            generate_text.append(next_token_id.item())

        res = self.model.tokenizer.decode(generate_text)
        return res

    def inference(self, representation, question):
        return self.decode(representation, prompt = question)

    def index(self, text):
        with torch.no_grad():
            representation =  self.encode(text)
        return torch.mean(representation, dim=-2).squeeze(0)

def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

class ICAEModel(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.icae = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16, attn_implementation="flash_attention_2", resume_download=True)
        
        self.training = self.model_args.train    
        
        if self.training:    # indepedent model for gradient checkpointing
            self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16, attn_implementation="flash_attention_2", resume_download=True)

        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2        

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3) 
        
        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=device).unsqueeze(0)    # mem tokens
        
        if self.training:
            self.init()


    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
        
    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))
        return num_segments


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # encoder part
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim)).to(prompt_answer_embs)
        
        for segment_idx in range(num_segments):
            
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(
                inputs_embeds=segment_input_embedding,
                output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
            
        # decoder part
        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens

        prompt_answer_embs[decoder_mem_flag] = compress_outputs  # replace memory slots
        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
        prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).to(prompt_answer_embs)    # replace special token's embedding from self.memory_token_embed
        
        if self.training:   # has an independent se.f.decoder
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():   # no independent decoder; use self.icae
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)


        logits = decoder_outputs.logits
        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
        target_ids = labels[:,1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target_ids)
        return {"loss": loss, "logits": logits}
    
    
    def tokens_to_embeddings(self, token_ids):   # input_tokens can be either normal tokens and special tokens
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings)    # replace special token's embedding from self.memory_token_embed
        return embeddings
        
    
    def _compress(
        self,
        input_ids: torch.LongTensor = None
    ):  # for inference; compress a fixed length of input into memory slots

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((max_compressed_length, self.dim))
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            # collect memory tokens
            compress_outputs[segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = segment_compress_outputs[mem_flag]
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
        
        return compress_outputs