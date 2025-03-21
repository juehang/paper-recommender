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
progress_tracker = None  # Global progress tracker
is_initialized = False   # Flag to track initialization status

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
            percentage = (self.current / self.total * 100) if self.total > 0 else 0
            # Print debug messages
            print(f"DEBUG: update_ui called - {self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
            
            # Call the properly exposed updateProgress function
            try:
                eel.updateProgress(self.current, self.total, self.description)
                print(f"DEBUG: Called updateProgress successfully")
            except Exception as e:
                print(f"DEBUG: Error calling updateProgress: {str(e)}")
        except Exception as e:
            # Print the exception for debugging
            print(f"DEBUG: Error in update_ui: {str(e)}")
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
    Only initializes once, subsequent calls will return immediately.
    """
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
    
    # Return if already initialized
    if is_initialized:
        return
    
    # Load configuration
    config = load_config()
    
    # Create global progress tracker
    progress_tracker = UiProgressTracker()
    
    # Create components
    data_source = ArXivDataSource(period=config["period_hours"])
    embedding_model = create_embedding_model(config)
    
    # Set progress tracker for embedding model
    embedding_model.set_progress_tracker(progress_tracker)
    
    # Create vector store
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Create onboarder using existing components
    _, _, _, onboarder = create_onboarding_system(
        period=config["period_hours"],
        random_sample_size=config["random_sample_size"],
        diverse_sample_size=config["diverse_sample_size"],
        chroma_db_path=config["chroma_db_path"],
        embedding_cache_path=config["embedding_cache_path"],
        config=config,
        data_source=data_source,
        embedding_model=embedding_model,
        vector_store=vector_store
    )
    
    # Create recommender
    recommender = create_recommender(
        data_source,
        vector_store,
        embedding_model,
        max_samples=config["max_samples"],
        model_path=config["model_path"]
    )
    
    # Set initialization flag
    is_initialized = True

# Helper function to reset the global progress tracker
def reset_progress_tracker(total=0, description="Processing"):
    """
    Reset the global progress tracker with new parameters.
    
    Args:
        total (int): Total number of steps
        description (str): Description of the operation
    
    Returns:
        UiProgressTracker: The reset progress tracker
    """
    global progress_tracker, is_initialized
    
    # Initialize components if not already initialized
    if not is_initialized:
        init_components()
    
    # Reset the progress tracker
    progress_tracker.reset(total=total, description=description)
    
    return progress_tracker

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
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
    if not is_initialized:
        init_components()
    elif config is None:
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
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
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
    global onboarder, is_initialized
    if not is_initialized:
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
    global onboarder, is_initialized
    if not is_initialized:
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
    global onboarder, is_initialized
    if not is_initialized:
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
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
    if not is_initialized:
        init_components()
    
    # Initialize the progress tracker with a description only
    # The actual total and progress updates will be handled by the recommender
    reset_progress_tracker(total=0, description="Starting recommendation process")
    
    try:
        # Use provided values or defaults from config
        num_recs = num_recommendations or config["num_recommendations"]
        exp_weight = exploration_weight or config["exploration_weight"]
        n_nearest = n_nearest_embeddings or config["n_nearest_embeddings"]
        num_samples = gp_num_samples or config["gp_num_samples"]
        
        # Make sure the embedding model has the global progress tracker
        embedding_model.set_progress_tracker(progress_tracker)
        
        # Get recommendations - the recommender will handle progress tracking
        recommendations = recommender.recommend(
            num_recommendations=num_recs,
            exploration_weight=exp_weight,
            n_nearest_embeddings=n_nearest,
            gp_num_samples=num_samples
        )
        
        # Update description for final processing
        progress_tracker.description = "Processing recommendation results"
        
        # Convert to JSON-serializable format
        result = []
        for rec in recommendations:
            # Create a copy without the embedding (not needed in UI)
            # but keep the document field which is needed for rating submissions
            rec_copy = {k: v for k, v in rec.items() if k != 'embedding'}
            result.append(rec_copy)
        
        # Complete the progress
        progress_tracker.description = "Recommendations complete"
        
        return result
    except Exception as e:
        # Update progress tracker to show error
        progress_tracker.description = f"Error generating recommendations: {str(e)}"
        # Re-raise the exception
        raise

@eel.expose
def add_recommendation_ratings(ratings: List[Dict[str, Any]]) -> int:
    """
    Add ratings for recommended papers.
    
    Args:
        ratings (list): List of dictionaries with document, link, and rating
        
    Returns:
        int: Number of ratings added
    """
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
    if not is_initialized:
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
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
    if not is_initialized:
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
    global vector_store, is_initialized
    if not is_initialized:
        init_components()
    
    # Reset the global progress tracker for this operation
    reset_progress_tracker(total=100, description="Loading database entries")
    progress_tracker.update_to(10)  # Show initial progress
    
    try:
        # Get documents filtered by time directly from the database
        progress_tracker.update_to(30)
        filtered_docs = vector_store.get_documents_by_time(days=time_filter)
        progress_tracker.update_to(50)
        
        # Convert to a more user-friendly format
        result = []
        total_docs = len(filtered_docs["ids"])
        
        # Update progress tracker with the actual number of documents
        progress_tracker.reset(total=total_docs + 50, description="Processing database entries")
        progress_tracker.update_to(50)  # Start at 50% after fetching data
        
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
            
            # Update progress every few documents to avoid too many updates
            if i % max(1, total_docs // 10) == 0:
                progress_tracker.update_to(50 + int((i / total_docs) * 50))
        
        # Complete the progress
        progress_tracker.update_to(progress_tracker.total)
        return result
    finally:
        # No need to close the progress tracker as it's reused
        pass

@eel.expose
def recompute_embeddings() -> int:
    """
    Recompute all embeddings in the vector store.
    
    This function:
    1. Clears the embedding cache
    2. Recomputes all embeddings in the vector store
    
    Returns:
        int: The number of documents recomputed
    """
    global vector_store, embedding_model, progress_tracker, is_initialized
    if not is_initialized:
        init_components()
    
    # Reset the global progress tracker for this operation
    reset_progress_tracker(total=100, description="Preparing to recompute embeddings")
    progress_tracker.update_to(10)  # Show initial progress
    
    try:
        # Recompute embeddings
        doc_count = vector_store.recompute_embeddings(progress_tracker=progress_tracker)
        
        # Complete the progress
        progress_tracker.description = f"Completed recomputing embeddings for {doc_count} documents"
        progress_tracker.update_to(progress_tracker.total)
        
        return doc_count
    except Exception as e:
        progress_tracker.description = f"Error recomputing embeddings: {str(e)}"
        raise

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
    global vector_store, embedding_model, is_initialized
    if not is_initialized:
        init_components()
    
    # Reset the global progress tracker for this operation
    reset_progress_tracker(total=100, description="Performing semantic search")
    progress_tracker.update_to(10)  # Show initial progress
    
    try:
        # Perform semantic search with time filtering at the database level
        progress_tracker.update_to(30)
        results = vector_store.search_by_time([query_text], num_results=num_results, days=time_filter)
        progress_tracker.update_to(50)
        
        # Process results
        processed_results = []
        
        if "distances" in results and results["distances"]:
            # Get all documents to retrieve the full document text
            progress_tracker.update_to(60)
            all_docs = vector_store.get_all_documents()
            progress_tracker.update_to(70)
            
            # Update progress tracker with the actual number of results
            total_results = len(results["distances"][0])
            progress_tracker.reset(total=total_results + 70, description="Processing search results")
            progress_tracker.update_to(70)  # Start at 70% after fetching data
            
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
                
                # Update progress
                progress_tracker.update_to(70 + int((i / total_results) * 30))
        
        # Complete the progress
        progress_tracker.update_to(progress_tracker.total)
        return processed_results
    finally:
        # No need to close the progress tracker as it's reused
        pass

@eel.expose
def get_gp_visualization_data(sample_size: int = 30) -> Dict[str, Any]:
    """
    Get data for Gaussian Process visualization.
    
    Args:
        sample_size (int): Number of random papers to include in visualization
        
    Returns:
        dict: Visualization data including points, labels, and metadata
    """
    global config, data_source, embedding_model, vector_store, onboarder, recommender, progress_tracker, is_initialized
    if not is_initialized:
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
            rating_diff = np.log((ratings[i] - ratings[j])**2+0.1)
            
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
    
    # Initialize all components
    init_components()
    
    # Start Eel
    eel.start('index.html', mode=mode, host=host, port=port, block=block)

def main():
    """
    Main entry point for the UI app.
    """
    start_app()

if __name__ == "__main__":
    main()
