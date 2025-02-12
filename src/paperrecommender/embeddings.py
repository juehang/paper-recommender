import requests
import numpy as np

class EmbeddingModel:
    """
    Base class for embedding models.
    """
    def generate_embedding(self, text, url, model):
        raise NotImplementedError("This method should be overridden by subclasses.")
    def get_embedding(self, text):
        return self.generate_embedding(text, self.url, self.model)

class OllamaEmbedding(EmbeddingModel):
    """
    Class for generating embeddings using the Ollama API.
    """
    def __init__(self, url="http://localhost:11434/api/embeddings", model="nomic-embed-text"):
        self.url = url
        self.model = model
    
    @staticmethod
    def generate_embedding(text, url, model):
        json_data = {"model": model, "prompt": text}
        response = requests.post(url, json=json_data, headers={"Content-Type": "application/json"})
        return np.array(response.json()['embedding'])