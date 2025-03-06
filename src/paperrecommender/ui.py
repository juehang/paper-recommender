import eel
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

from .config import load_config, save_config, is_first_startup
from .onboarding import create_onboarding_system, Onboarder
from .recommender import create_recommender, bootstrap_recommender as bootstrap_recommender_func
from .data_sources import ArXivDataSource
from .embeddings import create_embedding_model
from .vector_store import ChromaVectorStore
from .common import ProgressTracker

# Initialize global variables
config = None
data_source = None
embedding_model = None
vector_store = None
onboarder = None
recommender = None

# Custom progress tracker for UI
class UiProgressTracker(ProgressTracker):
    """
    Progress tracker that updates the UI via Eel.
    """
    def __init__(self, total=0, description="Processing"):
        super().__init__(total=total, description=description, disable=True)  # Disable tqdm output
        self.update_ui()
    
    def update(self, value=1):
        super().update(value)
        self.update_ui()
    
    def update_to(self, value):
        super().update_to(value)
        self.update_ui()
    
    def reset(self, total=None, description=None):
        super().reset(total, description)
        self.update_ui()
    
    def update_ui(self):
        try:
            eel.updateProgress(self.current, self.total, self.description)
        except Exception:
            # Ignore errors if Eel is not initialized yet
            pass

# Initialize Eel
def init_eel(web_dir=None):
    """
    Initialize Eel with the web directory.
    
    Args:
        web_dir (str, optional): Path to the web directory. If None, use the default.
    """
    if web_dir is None:
        # Use the web directory in the package
        web_dir = os.path.join(os.path.dirname(__file__), 'web')
    
    # Initialize Eel
    eel.init(web_dir)

# Initialize components
def init_components():
    """
    Initialize the components (data source, embedding model, vector store, etc.)
    """
    global config, data_source, embedding_model, vector_store, onboarder, recommender
    
    # Load configuration
    config = load_config()
    
    # Create components
    data_source = ArXivDataSource(period=config["period_hours"])
    embedding_model = create_embedding_model(config)
    
    # Set up progress tracker
    progress_tracker = UiProgressTracker()
    embedding_model.set_progress_tracker(progress_tracker)
    
    # Create vector store
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Create onboarder
    _, _, _, onboarder = create_onboarding_system(
        period=config["period_hours"],
        random_sample_size=config["random_sample_size"],
        diverse_sample_size=config["diverse_sample_size"],
        chroma_db_path=config["chroma_db_path"],
        embedding_cache_path=config["embedding_cache_path"],
        config=config
    )
    
    # Create recommender
    recommender = create_recommender(
        data_source, 
        vector_store, 
        embedding_model,
        max_samples=config["max_samples"],
        model_path=config["model_path"]
    )

# Eel exposed functions

@eel.expose
def is_first_startup() -> bool:
    """
    Check if this is the first startup.
    
    Returns:
        bool: True if this is the first startup, False otherwise
    """
    from .config import is_first_startup as config_is_first_startup
    return config_is_first_startup()

@eel.expose
def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        dict: The current configuration
    """
    global config
    if config is None:
        config = load_config()
    return config

@eel.expose
def save_config(new_config: Dict[str, Any]) -> bool:
    """
    Save the configuration.
    
    Args:
        new_config (dict): The new configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    global config
    try:
        save_config(new_config)
        config = new_config
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

@eel.expose
def prepare_onboarding_candidates() -> List[List[Dict[str, Any]]]:
    """
    Prepare candidate papers for onboarding.
    
    Returns:
        list: A list of lists of paper dictionaries, one list per strategy
    """
    global onboarder
    if onboarder is None:
        init_components()
    
    # Prepare candidates
    strategy_papers = onboarder.prepare_candidates()
    
    # Convert to JSON-serializable format
    result = []
    for papers in strategy_papers:
        strategy_result = []
        for paper in papers:
            # Create a copy without the embedding (not JSON serializable)
            paper_copy = {
                'title': paper['title'],
                'abstract': paper['abstract'],
                'link': paper['link'],
                'strategy_index': paper['strategy_index'],
                'rating': paper['rating']
            }
            strategy_result.append(paper_copy)
        result.append(strategy_result)
    
    return result

@eel.expose
def commit_onboarding(ratings: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    Commit the onboarding process with the provided ratings.
    
    Args:
        ratings (list): List of (index, rating) tuples
        
    Returns:
        dict: Statistics about the onboarding process
    """
    global onboarder
    if onboarder is None:
        init_components()
    
    # Set ratings
    onboarder.set_ratings_batch(ratings)
    
    # Commit onboarding
    return onboarder.commit_onboarding(default_rating=0, bootstrap_recommender=True)

@eel.expose
def add_custom_paper(title: str, abstract: str, link: str, rating: int) -> bool:
    """
    Add a custom paper.
    
    Args:
        title (str): The title of the paper
        abstract (str): The abstract of the paper
        link (str): The link to the paper
        rating (int): The rating of the paper
        
    Returns:
        bool: True if successful, False otherwise
    """
    global onboarder
    if onboarder is None:
        init_components()
    
    return onboarder.add_custom_paper(title, abstract, link, rating)

@eel.expose
def get_recommendations(
    num_recommendations: Optional[int] = None,
    exploration_weight: Optional[float] = None,
    n_nearest_embeddings: Optional[int] = None,
    gp_num_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get recommendations.
    
    Args:
        num_recommendations (int, optional): Number of recommendations to return
        exploration_weight (float, optional): Exploration weight
        n_nearest_embeddings (int, optional): Number of nearest embeddings to use
        gp_num_samples (int, optional): Number of GP samples
        
    Returns:
        list: A list of recommended papers
    """
    global recommender, config
    if recommender is None:
        init_components()
    
    # Use provided values or defaults from config
    num_recs = num_recommendations or config["num_recommendations"]
    exp_weight = exploration_weight or config["exploration_weight"]
    n_nearest = n_nearest_embeddings or config["n_nearest_embeddings"]
    num_samples = gp_num_samples or config["gp_num_samples"]
    
    # Get recommendations
    recommendations = recommender.recommend(
        num_recommendations=num_recs,
        exploration_weight=exp_weight,
        n_nearest_embeddings=n_nearest,
        gp_num_samples=num_samples
    )
    
    # Convert to JSON-serializable format
    result = []
    for rec in recommendations:
        # Create a copy without the document and embedding (not needed in UI)
        rec_copy = {k: v for k, v in rec.items() if k != 'document' and k != 'embedding'}
        result.append(rec_copy)
    
    return result

@eel.expose
def add_recommendation_ratings(ratings: List[Dict[str, Any]]) -> int:
    """
    Add ratings for recommended papers.
    
    Args:
        ratings (list): List of dictionaries with document, link, and rating
        
    Returns:
        int: Number of ratings added
    """
    global vector_store
    if vector_store is None:
        init_components()
    
    count = 0
    for rating_info in ratings:
        document = rating_info["document"]
        link = rating_info["link"]
        rating = rating_info["rating"]
        
        vector_store.add_document(document, link, rating)
        count += 1
    
    return count

@eel.expose
def bootstrap_recommender() -> bool:
    """
    Bootstrap the recommender.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global recommender
    if recommender is None:
        init_components()
    
    return recommender.bootstrap(force=True)

# Start the Eel app
def start_app(web_dir=None, mode="chrome-app", host="localhost", port=8000, block=True):
    """
    Start the Eel app.
    
    Args:
        web_dir (str, optional): Path to the web directory. If None, use the default.
        mode (str, optional): Mode to start Eel in. Default is "chrome-app".
        host (str, optional): Host to bind to. Default is "localhost".
        port (int, optional): Port to bind to. Default is 8000.
        block (bool, optional): Whether to block the main thread. Default is True.
    """
    # Initialize Eel
    init_eel(web_dir)
    
    # Load configuration only (defer heavy component initialization)
    global config
    if config is None:
        config = load_config()
    
    # Start Eel
    eel.start('index.html', mode=mode, host=host, port=port, block=block)

def main():
    """
    Main entry point for the UI app.
    """
    start_app()

if __name__ == "__main__":
    main()
