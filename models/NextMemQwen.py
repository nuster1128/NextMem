import os
from typing import Optional, Union
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer, AddedToken

from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel, Qwen3Model
from transformers.generation import GenerationMixin
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from peft import PeftModel

special_tokens = [
    '<|start_of_document|>',
    '<|end_of_memory|>'
]


special_tokens_initialize_map = {
    '<|start_of_document|>': 'repeat',
    '<|end_of_memory|>': 'end'
}

class ModifiedQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class NextMemQwen():
    def __init__(self, config):
        self.config = config

        self.model = ModifiedQwen3ForCausalLM.from_pretrained(
            config['model_path'],
            dtype=torch.bfloat16,
            device_map=config['device']
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_path']
        )

        self.special_input_embeddings = torch.nn.Embedding(len(special_tokens)+1, self.model.config.hidden_size, dtype=torch.bfloat16).to(self.model.device)

        self.tokenizer.add_tokens([
            AddedToken(token, lstrip=False, rstrip=False, normalized=False, special=False) for token in special_tokens
        ])

        with torch.no_grad():
            for token in special_tokens:
                init_token_ids = [self.tokenizer.convert_tokens_to_ids(word) for word in special_tokens_initialize_map[token].split()]
                
                input_init_embedding = self.model.get_input_embeddings().weight[init_token_ids].mean(dim=0)
                self.model.get_input_embeddings().weight[self.tokenizer.convert_tokens_to_ids(token)] = input_init_embedding
        
        stage_1_checkpoint = torch.load(config['stage_1_checkpoint_path']+'/special_tokens_DTOKENS.pt', map_location=self.model.device)
        self.special_input_embeddings.load_state_dict(stage_1_checkpoint['special_input_embeddings'])
        self.model = PeftModel.from_pretrained(self.model, config['stage_1_checkpoint_path'])
        self.model = self.model.merge_and_unload()

        stage_2_checkpoint_path =  config['stage_2_checkpoint_path']
        self.model = PeftModel.from_pretrained(self.model, stage_2_checkpoint_path)
    def input_ids_embeddings(self, input_ids):
        if not hasattr(self, 'special_ids_tensor'):
            self.special_ids_tensor = self.tokenizer(special_tokens, return_tensors="pt").input_ids.squeeze().to(self.model.device)

        original_embeddings = self.model.get_input_embeddings()(input_ids)            

        special_input_masks = torch.isin(input_ids, self.special_ids_tensor)
        special_input_ids = torch.where(special_input_masks, input_ids-self.special_ids_tensor[0], len(special_tokens))
        special_input_coef = torch.where(special_input_masks, 1.0, 0.0).unsqueeze(-1).to(torch.bfloat16)
        special_input_embeddings = self.special_input_embeddings(special_input_ids) * special_input_coef

        return original_embeddings + special_input_embeddings

    def get_encode_inputs(self, inputs):
        return {
            'inputs_embeds': self.input_ids_embeddings(inputs['input_ids']),
            'attention_mask': inputs['attention_mask'],
        }

    def batch_encode(self, batch_inputs, latent_length):
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

        # Add the [SoD] token.
        batch_inputs = self.get_encode_inputs({
            'input_ids': torch.cat([
                batch_inputs['input_ids'], 
                torch.ones_like(batch_inputs['input_ids'][:, -1:]) * self.tokenizer.convert_tokens_to_ids('<|start_of_document|>')
                ], dim=-1),
            'attention_mask': torch.cat([
                batch_inputs['attention_mask'], torch.ones_like(batch_inputs['attention_mask'][:, -1:])
                ], dim=-1),
        })

        batch_inputs_embeds = batch_inputs['inputs_embeds']
        batch_attention_masks = batch_inputs['attention_mask']

        # batch_inputs_embeds.requires_grad_(True)

        latent_representation_cache = []
        
        for idx in range(latent_length):
            
            outputs = self.model(
                inputs_embeds = batch_inputs_embeds,
                attention_mask = batch_attention_masks,
                user_cache = False,
                output_hidden_states=True,
            )

            next_embedding = outputs.hidden_states[-1][:,-1,:].unsqueeze(-2)
            latent_representation_cache.append(next_embedding)

            batch_inputs_embeds = torch.cat([batch_inputs_embeds, next_embedding.detach()], dim=-2)
            batch_attention_masks = torch.cat([batch_attention_masks, torch.ones_like(batch_attention_masks[:, -1:])], dim=-1)

        latent_representations = torch.cat(latent_representation_cache, dim=-2)
        latent_attention_mask = torch.ones(latent_representations.shape[:-1]).to(batch_attention_masks)
        
        return latent_representations, latent_attention_mask

    def batch_decode(self, latent_representations, latent_attention_mask, suffix_inputs=None):
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        
        decode_suffix_embeddings= self.input_ids_embeddings(suffix_inputs['input_ids'])
        decode_suffix_attention_mask = suffix_inputs['attention_mask']
            
        inputs_embeds = torch.cat([latent_representations, decode_suffix_embeddings], dim=-2)
        inputs_attention_mask = torch.cat([latent_attention_mask, decode_suffix_attention_mask], dim=-1)

        with self.model.disable_adapter():
            outputs = self.model(
                inputs_embeds = inputs_embeds,
                attention_mask = inputs_attention_mask,
                ## use_cache=True,
            )

        return outputs

    def encode(self, text, latent_length=None):
        if not latent_length:
            latent_length = self.config['max_latent_length']
        inputs = self.tokenizer([text], return_tensors="pt", max_length=self.config['max_encode_length'], 
                     add_special_tokens=False, truncation=True, padding=False).to(self.model.device)
        with torch.no_grad():
            latent_representations, latent_attention_mask = self.batch_encode(inputs,latent_length)
        return latent_representations

    def decode(self, representation, prompt = '<|start_of_document|>'):
        suffix_inputs = self.tokenizer([prompt], return_tensors="pt", max_length=self.config['max_length'], 
                     add_special_tokens=False, truncation=True, padding=False).to(self.model.device)

        inputs = {
            'inputs_embeds': torch.cat([
                representation,
                self.input_ids_embeddings(suffix_inputs['input_ids'])
            ], dim=-2),
            'attention_mask': torch.cat([
                torch.ones(representation.shape[:-1]).to(suffix_inputs['attention_mask']),
                suffix_inputs['attention_mask']
            ], dim=-1),
        }
        with self.model.disable_adapter():
            outputs = self.model.generate(
                inputs_embeds=inputs['inputs_embeds'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.config['max_length'],
            )
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return res

    def inference(self, latent_representations, question):
        instructed_prompt = self.tokenizer.apply_chat_template(
                [  
                    {"role": "system", "content": ""},
                    {"role": "system", "content": "Briefly answer the following question."},
                    {"role": "user", "content": f"Question: {question} Answer:"}
                    ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

        suffix_inputs = self.tokenizer([instructed_prompt], return_tensors="pt", max_length=self.config['max_length'], 
                add_special_tokens=False, truncation=True, padding=False).to(self.model.device)
        

        latent_attention_mask = torch.ones(latent_representations.shape[:-1]).to(suffix_inputs['attention_mask'])
        
        suffix_input_ids = suffix_inputs['input_ids']
        suffix_attention_mask = suffix_inputs['attention_mask']

        # for idx, sid in enumerate(suffix_input_ids[0]):
        #     print(idx, sid, self.tokenizer.decode(sid))

        sod_inputs = {
                'input_ids': torch.tensor([self.tokenizer.convert_tokens_to_ids('<|start_of_document|>')], dtype=torch.int32
                ).expand(suffix_input_ids.shape[0], -1).to(self.model.device),
                'attention_mask': torch.ones((suffix_input_ids.shape[0], 1)).to(suffix_attention_mask)
            }
        sod_representations = self.input_ids_embeddings(sod_inputs['input_ids'])
        eod_inputs = {
            'input_ids': torch.tensor([self.tokenizer.convert_tokens_to_ids('<|endoftext|>')], dtype=torch.int32
            ).expand(suffix_input_ids.shape[0], -1).to(self.model.device),
            'attention_mask': torch.ones((suffix_input_ids.shape[0], 1)).to(suffix_attention_mask)
        }
        eod_representations = self.input_ids_embeddings(eod_inputs['input_ids'])

        inputs_embeds = torch.cat([
            self.input_ids_embeddings(suffix_input_ids[..., :2, ...]),
            sod_representations,
            latent_representations,
            eod_representations,
            self.input_ids_embeddings(suffix_input_ids[..., 2:, ...]),
            ], dim=-2)
        
        inputs_attention_mask = torch.cat([
            suffix_attention_mask[..., :2],
            sod_inputs['attention_mask'],
            latent_attention_mask,
            eod_inputs['attention_mask'],
            suffix_attention_mask[..., 2:]], dim=-1)

        with self.model.disable_adapter():
            outputs = self.model.generate(
                inputs_embeds = inputs_embeds,
                attention_mask = inputs_attention_mask,
                max_new_tokens=1024,
                ## use_cache=True,
            )
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        if 'Answer:' in res:
            if len(res.split('Answer:')[-1].strip()) > 0:
                res = res.split('Answer:')[-1].strip()
            else:
                res = ''.join(res.split('Answer:')[:-1])

        return res
    
    def index(self, text, latent_length=None):
        if len(text) == 0:
            text = 'None'
        with torch.no_grad():
            representation =  self.encode(text, latent_length)
        return torch.mean(representation, dim=-2).squeeze(0)
