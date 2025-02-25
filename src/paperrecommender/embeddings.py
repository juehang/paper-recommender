import requests
import numpy as np

class EmbeddingModel:
    """
    Base class for embedding models.
    """
    def __init__(self, max_cache_size=1000):
        self.progress_tracker = None
        self.embedding_cache = {}  # Cache for storing embeddings
        self.max_cache_size = max_cache_size
        
    def set_progress_tracker(self, tracker):
        """Set a progress tracker for this embedding model"""
        self.progress_tracker = tracker
        
    def _manage_cache_size(self):
        """Ensure the cache doesn't exceed the maximum size"""
        if len(self.embedding_cache) > self.max_cache_size:
            # Remove oldest entries (first inserted)
            items_to_remove = len(self.embedding_cache) - self.max_cache_size
            for key in list(self.embedding_cache.keys())[:items_to_remove]:
                del self.embedding_cache[key]
        
    def generate_embedding(self, text, url, model):
        raise NotImplementedError("This method should be overridden by subclasses.")
        
    def get_embedding(self, texts):
        # Handle both single text and list of texts
        if isinstance(texts, str):
            # Check if we have this text in the cache
            if texts in self.embedding_cache:
                # Return cached embedding without updating progress
                return self.embedding_cache[texts]
                
            # Generate new embedding
            result = self.generate_embedding(texts, self.url, self.model)
            
            # Cache the result
            self.embedding_cache[texts] = result
            
            # Manage cache size
            self._manage_cache_size()
            
            # Update progress if tracker exists
            if self.progress_tracker:
                self.progress_tracker.update()
                
            return result
        else:
            # For lists, we'll process each item individually
            # If we have a progress tracker, update its total to reflect only uncached items
            if self.progress_tracker:
                # Count how many items are not in cache
                uncached_count = sum(1 for item in texts if item not in self.embedding_cache)
                self.progress_tracker.reset(total=uncached_count)
                
            results = []
            for item in texts:
                # Check cache first
                if item in self.embedding_cache:
                    results.append(self.embedding_cache[item])
                else:
                    # Generate and cache new embedding
                    result = self.generate_embedding(item, self.url, self.model)
                    self.embedding_cache[item] = result
                    results.append(result)
                    
                    # Manage cache size
                    self._manage_cache_size()
                    
                    # Update progress only for newly generated embeddings
                    if self.progress_tracker:
                        self.progress_tracker.update()
                        
            return results

class OllamaEmbedding(EmbeddingModel):
    """
    Class for generating embeddings using the Ollama API.
    """
    def __init__(self, url="http://localhost:11434/api/embeddings", model="nomic-embed-text", max_cache_size=1000):
        super().__init__(max_cache_size=max_cache_size)
        self.url = url
        self.model = model
    
    @staticmethod
    def generate_embedding(text, url, model):
        json_data = {"model": model, "prompt": text}
        response = requests.post(url, json=json_data, headers={"Content-Type": "application/json"})
        return np.array(response.json()['embedding'])
