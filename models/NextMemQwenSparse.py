from models.NextMemQwen import NextMemQwen
import torch

NF4_VALUES = torch.tensor([
    -1.0, -0.6961928, -0.52507305, -0.3949174, -0.28444138, -0.18477343, 
    -0.091050036, 0.0, 0.079580299, 0.1609302, 0.2461123, 0.33791524, 
    0.44070982, 0.562617, 0.72295684, 1.0
], dtype=torch.bfloat16)

class NextMemQwenSparse():
    def __init__(self, config):
        self.nextmem_dense = NextMemQwen(config)

    def quantize(self, representation):
        representation = representation[0,...]
        scales = torch.max(torch.abs(representation), dim=0, keepdim=True)[0]
        representation_norm = representation / (scales + 1e-12)
        dists = torch.abs(representation_norm.unsqueeze(-1) - NF4_VALUES.to(representation.device))
        quant_indices = torch.argmin(dists, dim=-1).to(torch.uint8)
        
        return quant_indices, scales.to(torch.float8_e4m3fn)

    def dequantize(self, representation):
        quant_indices, scales = representation
        nf4_map = NF4_VALUES.to(scales.device)
        x_dequant = nf4_map[quant_indices.long()]
        
        return (x_dequant * scales.to(torch.bfloat16)).to(self.nextmem_dense.model.device).unsqueeze(0)
    
    def encode(self, text, latent_length=None):
        representation = self.nextmem_dense.encode(text, latent_length)
        quantized_representation = self.quantize(representation)
        return quantized_representation
    
    def decode(self, representation, prompt = '<|start_of_document|>'):
        dequantized_representation = self.dequantize(representation)
        text = self.nextmem_dense.decode(dequantized_representation, prompt)
        return text

    def inference(self, representation, question):
        latent_representations = self.dequantize(representation)

        return self.nextmem_dense.inference(latent_representations, question)

    def index(self, text, latent_length=None):
        if len(text) == 0:
            text = 'None'
        with torch.no_grad():
            representation =  self.encode(text, latent_length)
            representation = self.dequantize(representation)
        return torch.mean(representation, dim=-2).squeeze(0)