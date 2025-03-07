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
from .common import ProgressTracker, cosine_similarity

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
        print("Loading configuration...")
        config = load_config()
        print(f"Configuration loaded: {len(config)} keys")
    
    # Create a clean copy of the config that's guaranteed to be JSON-serializable
    json_safe_config = {}
    for key, value in config.items():
        # Convert Path objects to strings
        if isinstance(value, (Path, os.PathLike)):
            json_safe_config[key] = str(value)
        else:
            json_safe_config[key] = value
    
    print(f"Returning JSON-safe configuration with {len(json_safe_config)} keys")
    return json_safe_config

@eel.expose
def save_ui_config(new_config: Dict[str, Any]) -> bool:
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
        # Create a copy without the embedding (not needed in UI)
        # but keep the document field which is needed for rating submissions
        rec_copy = {k: v for k, v in rec.items() if k != 'embedding'}
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

@eel.expose
def get_chroma_documents(time_filter=30) -> List[Dict[str, Any]]:
    """
    Get documents from the ChromaDB vector store with optional time filtering.
    
    Args:
        time_filter (int): Number of days to filter by (0 for all documents)
        
    Returns:
        list: A list of documents with their metadata
    """
    global vector_store
    if vector_store is None:
        init_components()
    
    # Get documents filtered by time directly from the database
    filtered_docs = vector_store.get_documents_by_time(days=time_filter)
    
    # Convert to a more user-friendly format
    result = []
    for i, (doc_id, document, metadata) in enumerate(zip(
        filtered_docs["ids"], 
        filtered_docs["documents"], 
        filtered_docs["metadatas"]
    )):
        # Format timestamp as human-readable if it exists
        timestamp_display = "N/A"
        if "timestamp" in metadata:
            try:
                # Convert Unix timestamp to readable format
                from datetime import datetime
                timestamp = metadata["timestamp"]
                timestamp_display = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            except:
                timestamp_display = str(metadata["timestamp"])
        
        result.append({
            "id": doc_id,
            "document": document,
            "link": metadata.get("link", ""),
            "rating": metadata.get("rating", 0),
            "timestamp": metadata.get("timestamp", None),
            "timestamp_display": timestamp_display
        })
    
    return result

@eel.expose
def search_chroma_documents(query_text: str, num_results: int = 10, time_filter: int = 30) -> List[Dict[str, Any]]:
    """
    Perform semantic search on documents in the ChromaDB vector store.
    
    Args:
        query_text (str): The search query text
        num_results (int): Maximum number of results to return
        time_filter (int): Number of days to filter by (0 for all documents)
        
    Returns:
        list: A list of documents with their metadata and similarity scores
    """
    global vector_store, embedding_model
    if vector_store is None or embedding_model is None:
        init_components()
    
    # Perform semantic search with time filtering at the database level
    results = vector_store.search_by_time([query_text], num_results=num_results, days=time_filter)
    
    # Process results
    processed_results = []
    
    if "distances" in results and results["distances"]:
        # Get all documents to retrieve the full document text
        all_docs = vector_store.get_all_documents()
        
        for i, (distance, metadata) in enumerate(zip(results["distances"][0], results["metadatas"][0])):
            # Format timestamp as human-readable if it exists
            timestamp_display = "N/A"
            if "timestamp" in metadata:
                try:
                    # Convert Unix timestamp to readable format
                    from datetime import datetime
                    timestamp = metadata["timestamp"]
                    timestamp_display = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    timestamp_display = str(metadata["timestamp"])
            
            # Get the document text
            # Note: We need to retrieve the document text since it's not included in search results
            doc_index = -1
            for j, doc_id in enumerate(all_docs["ids"]):
                if doc_id == results["ids"][0][i]:
                    doc_index = j
                    break
            
            document = all_docs["documents"][doc_index] if doc_index >= 0 else "Document text not available"
            
            # Calculate similarity score (1 - distance)
            similarity = 1 - distance
            
            processed_results.append({
                "id": results["ids"][0][i],
                "document": document,
                "link": metadata.get("link", ""),
                "rating": metadata.get("rating", 0),
                "timestamp": metadata.get("timestamp", None),
                "timestamp_display": timestamp_display,
                "similarity": similarity
            })
    
    return processed_results

@eel.expose
def get_gp_visualization_data(sample_size: int = 30) -> Dict[str, Any]:
    """
    Get data for Gaussian Process visualization.
    
    Args:
        sample_size (int): Number of random papers to include in visualization
        
    Returns:
        dict: Visualization data including points, labels, and metadata
    """
    global recommender, vector_store
    if recommender is None or vector_store is None:
        init_components()
    
    # Ensure the recommender is bootstrapped
    if recommender.model is None:
        recommender.bootstrap()
    
    # Get all documents from the vector store
    all_docs = vector_store.get_all_documents()
    
    # Check if we have any documents
    if len(all_docs["ids"]) == 0:
        return {"points": [], "title": "No data available"}
    
    # Select a random sample of documents
    import numpy as np
    sample_size = min(sample_size, len(all_docs["ids"]))
    sample_indices = np.random.choice(len(all_docs["ids"]), sample_size, replace=False)
    
    # Prepare data for visualization
    doc_ids = [all_docs["ids"][i] for i in sample_indices]
    documents = [all_docs["documents"][i] for i in sample_indices]
    metadatas = [all_docs["metadatas"][i] for i in sample_indices]
    embeddings = [all_docs["embeddings"][i] for i in sample_indices]
    
    # Extract ratings and titles
    ratings = []
    titles = []
    for document, metadata in zip(documents, metadatas):
        # Get rating (default to 0 if not rated)
        rating = metadata.get("rating", 0)
        ratings.append(rating)
        
        # Extract title from document (first line)
        title = document.split('\n')[0][:50]
        titles.append(title)
    
    # Compute similarity matrix between documents
    similarity_matrix = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
        # Get embedding for document i
        embedding_i = embeddings[i]
        if embedding_i is None:
            continue
            
        for j in range(i+1, sample_size):
            # Get embedding for document j
            embedding_j = embeddings[j]
            if embedding_j is None:
                continue
                
            # Compute cosine similarity
            similarity = cosine_similarity(embedding_i, embedding_j)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Prepare data for GP model
    X = []  # Similarity scores
    y = []  # Rating variances
    
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            similarity = similarity_matrix[i, j]
            rating_diff = (ratings[i] - ratings[j])**2
            
            X.append([similarity])
            y.append(rating_diff)
    
    # Check if we have enough data points
    if len(X) < 10:
        return {"points": [], "title": "Not enough data points for visualization"}
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Get predictions from GP model
    if recommender.model is not None:
        # Generate a range of similarity values for prediction
        sim_range = np.linspace(0, 1, 100).reshape(-1, 1)
        
        # Get predictions
        mean_predictions, std_predictions = recommender.model.predict(sim_range, return_std=True)
        
        # Prepare points for visualization
        points = []
        
        # Add actual data points
        for i, (similarity, rating_diff) in enumerate(zip(X, y)):
            points.append({
                "x": float(similarity[0]),
                "y": float(rating_diff),
                "size": 5,
                "color": "#3498db",  # Blue for actual data
                "category": "Actual Data",
                "label": None
            })
        
        # Add prediction curve
        for i, (sim, pred, std) in enumerate(zip(sim_range, mean_predictions, std_predictions)):
            points.append({
                "x": float(sim[0]),
                "y": float(pred),
                "size": float(std),  # Size based on uncertainty
                "color": "#e74c3c",  # Red for predictions
                "category": "GP Prediction",
                "label": None
            })
        
        # Return visualization data
        return {
            "points": points,
            "title": "Gaussian Process Model: Rating Difference vs. Similarity",
            "xLabel": "Similarity Score",
            "yLabel": "Rating Difference"
        }
    else:
        # If no model is available, just return the raw data
        points = []
        
        for i, (similarity, rating_diff) in enumerate(zip(X, y)):
            points.append({
                "x": float(similarity[0]),
                "y": float(rating_diff),
                "size": 5,
                "color": "#3498db",
                "category": "Data Point",
                "label": None
            })
        
        return {
            "points": points,
            "title": "Rating Difference vs. Similarity (No GP Model)",
            "xLabel": "Similarity Score",
            "yLabel": "Rating Difference"
        }

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
