import numpy as np
import tqdm

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (list or numpy array): The first vector.
        vec2 (list or numpy array): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
        If the vectors are zero vectors, raises a ValueError.
    """
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError("Cannot compute cosine similarity for zero vectors.")
    else:
        return dot_product / (norm_vec1 * norm_vec2)
    
def construct_string(title, abstract):
    """
    Construct a single string from the title and abstract of a paper.
    Args:
        title (str): The title of the paper.
        abstract (str): The abstract of the paper.
        Returns:
           str: A combined string of the title and abstract.
    """
    return f"Title: {title}\nAbstract: {abstract}"

class ProgressTracker:
    """
    A flexible progress tracker that can be used programmatically by different UI implementations.
    Uses tqdm for efficient progress display.
    """
    def __init__(self, total=0, description="Processing", disable=False):
        """
        Initialize a progress tracker.
        
        Args:
            total (int): Total number of items to process
            description (str): Description of the progress operation
            disable (bool): Whether to disable the progress display
        """
        self.total = total
        self.current = 0
        self.description = description
        self.disable = disable
        self.tqdm_instance = None
        self.callbacks = []
        self._initialize_tqdm()
    
    def _initialize_tqdm(self):
        """Initialize or reinitialize the tqdm instance"""
        if self.tqdm_instance:
            self.tqdm_instance.close()
        
        self.tqdm_instance = tqdm.tqdm(
            total=self.total,
            desc=self.description,
            disable=self.disable
        )
    
    def update(self, value=1):
        """
        Update the progress.
        
        Args:
            value (int/float): Amount to increment progress by
        """
        if value <= 0:
            return
            
        self.current += value
        self.tqdm_instance.update(value)
        self._notify_callbacks()
    
    def update_to(self, value):
        """
        Update progress to a specific absolute value.
        
        Args:
            value (int/float): Absolute progress value to set
        """
        if value < self.current:
            return
            
        increment = value - self.current
        self.update(increment)
    
    def reset(self, total=None, description=None):
        """
        Reset the progress tracker.
        
        Args:
            total (int, optional): New total value
            description (str, optional): New description
        """
        self.current = 0
        if total is not None:
            self.total = total
        if description is not None:
            self.description = description
            
        self._initialize_tqdm()
        self._notify_callbacks()
    
    def register_callback(self, callback):
        """
        Register a callback function that will be called on updates.
        
        Args:
            callback: Function that takes (current, total, description)
        """
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks of the current progress"""
        for callback in self.callbacks:
            callback(self.current, self.total, self.description)
    
    def close(self):
        """Close the progress tracker and its tqdm instance"""
        if self.tqdm_instance:
            self.tqdm_instance.close()
