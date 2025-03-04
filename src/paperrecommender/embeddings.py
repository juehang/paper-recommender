import requests
import numpy as np
import os
import pickle
from openai import OpenAI

class EmbeddingModel:
    """
    Base class for embedding models.
    """
    def __init__(self, max_cache_size=1000, cache_path=None):
        self.progress_tracker = None
        self.embedding_cache = {}  # Cache for storing embeddings
        self.max_cache_size = max_cache_size
        self.cache_path = cache_path
        
        # Load cache from disk if it exists
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} embeddings from cache at {cache_path}")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
        
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
    
    def _save_cache(self):
        """Save the cache to disk if a cache path is set"""
        if self.cache_path:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
            except Exception as e:
                print(f"Error saving embedding cache: {e}")
        
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
            
            # Save cache to disk
            self._save_cache()
            
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
                    
                    # Save cache to disk
                    self._save_cache()
                    
                    # Update progress only for newly generated embeddings
                    if self.progress_tracker:
                        self.progress_tracker.update()
                        
            return results

class OllamaEmbedding(EmbeddingModel):
    """
    Class for generating embeddings using the Ollama API.
    """
    def __init__(self, url="http://localhost:11434/api/embeddings", model="nomic-embed-text", 
                 max_cache_size=2000, cache_path=None):
        super().__init__(max_cache_size=max_cache_size, cache_path=cache_path)
        self.url = url
        self.model = model
    
    @staticmethod
    def generate_embedding(text, url, model):
        json_data = {"model": model, "prompt": text}
        response = requests.post(url, json=json_data, headers={"Content-Type": "application/json"})
        return np.array(response.json()['embedding'])

class OpenAIEmbedding(EmbeddingModel):
    """
    Class for generating embeddings using the OpenAI API.
    """
    def __init__(self, api_key, model="text-embedding-ada-002", 
                 max_cache_size=2000, cache_path=None):
        super().__init__(max_cache_size=max_cache_size, cache_path=cache_path)
        self.client = OpenAI(api_key=api_key)
        self.model = model
        # Set url to None as it's not used in this implementation
        self.url = None
    
    @staticmethod
    def generate_embedding(text, url, model):
        # In this implementation, url is actually the OpenAI client
        client = url
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return np.array(response.data[0].embedding)


def create_embedding_model(config):
    """
    Create an embedding model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        EmbeddingModel: An instance of the configured embedding model
    """
    provider = config.get("embedding_provider", "ollama").lower()
    
    if provider == "openai":
        api_key = config.get("openai_api_key", "")
        if not api_key:
            print("Warning: OpenAI embedding provider selected but no API key provided. Falling back to Ollama.")
            return OllamaEmbedding(cache_path=config.get("embedding_cache_path"))
            
        return OpenAIEmbedding(
            api_key=api_key,
            model=config.get("openai_embedding_model", "text-embedding-ada-002"),
            cache_path=config.get("embedding_cache_path")
        )
    else:  # Default to Ollama
        return OllamaEmbedding(cache_path=config.get("embedding_cache_path"))
