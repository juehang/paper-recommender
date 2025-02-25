import requests
import numpy as np

class EmbeddingModel:
    """
    Base class for embedding models.
    """
    def __init__(self):
        self.progress_tracker = None
        
    def set_progress_tracker(self, tracker):
        """Set a progress tracker for this embedding model"""
        self.progress_tracker = tracker
        
    def generate_embedding(self, text, url, model):
        raise NotImplementedError("This method should be overridden by subclasses.")
        
    def get_embedding(self, texts):
        # Handle both single text and list of texts
        if isinstance(texts, str):
            result = self.generate_embedding(texts, self.url, self.model)
            if self.progress_tracker:
                self.progress_tracker.update()
            return result
        else:
            # If we have a progress tracker, update its total
            if self.progress_tracker:
                self.progress_tracker.reset(total=len(texts))
                
            results = []
            for item in texts:
                result = self.generate_embedding(item, self.url, self.model)
                results.append(result)
                if self.progress_tracker:
                    self.progress_tracker.update()
            return results

class OllamaEmbedding(EmbeddingModel):
    """
    Class for generating embeddings using the Ollama API.
    """
    def __init__(self, url="http://localhost:11434/api/embeddings", model="nomic-embed-text"):
        super().__init__()
        self.url = url
        self.model = model
    
    @staticmethod
    def generate_embedding(text, url, model):
        json_data = {"model": model, "prompt": text}
        response = requests.post(url, json=json_data, headers={"Content-Type": "application/json"})
        return np.array(response.json()['embedding'])
