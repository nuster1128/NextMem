from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from PIL import Image, ImageDraw, ImageFont
import time, os
import numpy as np
import torch

def text_wrap(text, font, max_width):
    lines = []
    if font.getlength(text) <= max_width:
        return [text]

    words = text.split(' ')
    current_line = ""
    
    for word in words:
        temp_line = current_line + ' ' + word if current_line else word
        
        if font.getlength(temp_line) <= max_width:
            current_line = temp_line
        else:
            lines.append(current_line)
            current_line = word
    
    lines.append(current_line)
    return lines

class DeepSeekOCR():
    def __init__(self, config):
        self.config = config
        self.img_size = config['img_size']

        self.model = AutoModel.from_pretrained(
            config['model_path'],
            _attn_implementation='flash_attention_2',
            device_map='cuda',
            trust_remote_code=True,
            use_safetensors=True
        ).eval().cuda().to(torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_path'],
            trust_remote_code=True
        )

    def create_img(self, reference):
        padding = 5
        font_size = 12
        img = Image.new('RGB', (self.img_size, self.img_size), color='white')
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype('~/.fonts/TIMES.TTF', int(font_size))
        max_text_width = self.img_size - 2 * padding
        wrapped_lines = text_wrap(reference, font, max_text_width)
        final_text = "\n".join(wrapped_lines)

        draw.text((padding, padding), final_text, fill='black', font=font)

        cache_path = f'vision_{int(time.time())}.png'

        img.save(cache_path)
        return cache_path
    
    def encode(self, text):
        img_path = self.create_img(text)
        return img_path

    def decode(self, representation):
        prompt = '<image>\nOCR this image.'
        res = self.model.infer(self.tokenizer, prompt=prompt, image_file=representation, output_path = '.', base_size = self.img_size, image_size = self.img_size, crop_mode=False, eval_mode=True)

        os.remove(representation)
        return res
    
    def inference(self, latent_representations, question):
        prompt = f'<image>\n{question}'
        res = self.model.infer(self.tokenizer, prompt=prompt, image_file=latent_representations, output_path = '.', base_size = self.img_size, image_size = self.img_size, crop_mode=False, eval_mode=True)

        os.remove(latent_representations)
        return res

    def index(self, text):
        img_path = self.create_img(text)
        
        prompt = '<image>\nOCR this image.'
        with torch.no_grad():
            representation = self.model.encode(self.tokenizer, prompt=prompt, image_file=img_path, output_path = '.', base_size = self.img_size, image_size = self.img_size, crop_mode=False, eval_mode=True)

        os.remove(img_path)
        return torch.mean(representation, dim=-2).squeeze(0)