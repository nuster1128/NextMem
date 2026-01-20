from sentence_transformers import SentenceTransformer

class BGE():
    def __init__(self, config):
        self.model = SentenceTransformer(config['model_path'])

    def index(self, text):
        representation = self.model.encode([text], convert_to_tensor=True)
        
        return representation.squeeze()