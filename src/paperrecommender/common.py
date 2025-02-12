import numpy as np

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