import os
import pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .common import construct_string

class Recommender:
    """
    Base class for recommender systems.
    """
    def __init__(self, data_source, vector_store, embedding_model):
        self.data_source = data_source
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
    def recommend(self, num_recommendations=10):
        """
        Generate recommendations.
        
        Args:
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: A list of recommended papers
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class GaussianRegressionRecommender(Recommender):
    """
    Recommender that uses 1D Gaussian Process Regression on similarity scores
    to predict paper ratings for new papers from a data source.
    """
    def __init__(self, data_source, vector_store, embedding_model, max_samples=50, model_path=None):
        super().__init__(data_source, vector_store, embedding_model)
        self.max_samples = max_samples
        self.model = None
        self.model_path = model_path or os.path.expanduser("~/.paper_recommender/gp_model.pkl")
        
        # Try to load an existing model
        self.load_model()
    
    def load_model(self):
        """
        Load a previously saved model if it exists.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False
    
    def save_model(self):
        """
        Save the current model to disk.
        
        Returns:
            bool: True if model was saved successfully, False otherwise
        """
        if self.model is None:
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
        return False
    
    def _get_training_data(self):
        """
        Get training data for the Gaussian Process Regression model.
        
        Returns:
            tuple: (X, y) where X is similarity scores and y is ratings
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get_all_documents()
        
        # If we don't have enough data, we can't fit the model
        if len(all_docs["ids"]) < 5:  # Arbitrary minimum threshold
            return None, None
            
        # We'll use each document in the vector store as a query
        # and collect similarity scores and ratings
        X = []  # Similarity scores
        y = []  # Ratings
        
        # Sample a subset of documents to use as queries
        # This prevents the bootstrapping process from being too slow
        sample_size = min(20, len(all_docs["ids"]))
        sample_indices = np.random.choice(len(all_docs["ids"]), sample_size, replace=False)
        
        for idx in sample_indices:
            # Use the document as a query
            query_text = all_docs["documents"][idx]
            results = self.vector_store.search([query_text], num_results=self.max_samples)
            
            if "distances" in results and results["distances"]:
                # Convert distances to similarities
                similarities = [1 - dist for dist in results["distances"][0]]
                ratings = [float(meta.get("rating", 0)) for meta in results["metadatas"][0]]
                
                X.extend(similarities)
                y.extend(ratings)
        
        # If we don't have enough data points, we can't fit the model
        if len(X) < 10:  # Arbitrary minimum threshold
            return None, None
            
        # Reshape X for scikit-learn (n_samples, n_features)
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        return X, y
        
    def bootstrap(self, force=False):
        """
        Bootstrap the Gaussian Process Regression model using existing entries in the database.
        
        Args:
            force (bool): If True, force retraining even if a model already exists
            
        Returns:
            bool: True if the model was successfully bootstrapped, False otherwise
        """
        # If we already have a model and aren't forcing a retrain, skip
        if self.model is not None and not force:
            return True
            
        X, y = self._get_training_data()
        
        if X is None or y is None:
            return False
            
        # Fit the Gaussian Process Regression model
        # Define the kernel with hyperparameters
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # Create and fit the model
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.model.fit(X, y)
        
        # Save the model
        self.save_model()
        
        return True
    
    def recommend(self, num_recommendations=10):
        """
        Recommend new papers from the data source.
        
        Args:
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: A list of recommended papers with predicted ratings
        """
        # Ensure we have fresh data
        self.data_source.refresh_data()
        
        # Check if we have a model
        if self.model is None:
            # Try to bootstrap if we don't have a model
            success = self.bootstrap()
            if not success:
                # If we couldn't bootstrap, return an empty list
                return []
        
        # Get new papers from the data source
        titles = self.data_source.titles
        abstracts = self.data_source.abstracts
        links = self.data_source.links
        
        if not titles:
            return []
        
        # Generate embeddings for new papers
        new_papers = []
        for title, abstract, link in zip(titles, abstracts, links):
            document = construct_string(title, abstract)
            embedding = self.embedding_model.get_embedding(document)
            
            new_papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "document": document,
                "embedding": embedding
            })
        
        # For each new paper, find similar papers in the vector store
        recommendations = []
        
        for paper in new_papers:
            # Use the paper as a query
            results = self.vector_store.search([paper["document"]], num_results=self.max_samples)
            
            if "distances" not in results or not results["distances"]:
                continue
                
            # Convert distances to similarities
            similarities = [1 - dist for dist in results["distances"][0]]
            
            # Reshape for prediction
            X_pred = np.array(similarities).reshape(-1, 1)
            
            # Predict rating
            predicted_rating, std_dev = self.model.predict(X_pred.mean(axis=0).reshape(1, -1), return_std=True)
            
            # Add to recommendations
            recommendations.append({
                "title": paper["title"],
                "abstract": paper["abstract"],
                "link": paper["link"],
                "document": paper["document"],
                "predicted_rating": float(predicted_rating[0]),
                "confidence": 1.0 - float(std_dev[0]) / 5.0  # Normalize to 0-1 range
            })
        
        # Sort by predicted rating (descending)
        recommendations.sort(key=lambda x: x["predicted_rating"], reverse=True)
        
        # Return top N recommendations
        return recommendations[:num_recommendations]


def create_recommender(data_source, vector_store, embedding_model, max_samples=50):
    """
    Create and initialize a recommender.
    
    Args:
        data_source: The data source to use for new papers
        vector_store: The vector store containing user-rated papers
        embedding_model: The embedding model to use for new papers
        max_samples: Maximum number of samples to use for fitting the model
        
    Returns:
        GaussianRegressionRecommender: An initialized recommender
    """
    recommender = GaussianRegressionRecommender(
        data_source, vector_store, embedding_model, max_samples
    )
    return recommender


def bootstrap_recommender():
    """
    Run a simple CLI interface for bootstrapping the recommender.
    """
    from .data_sources import ArXivDataSource
    from .embeddings import OllamaEmbedding
    from .vector_store import ChromaVectorStore
    
    print("\n===== BOOTSTRAPPING RECOMMENDATION SYSTEM =====\n")
    
    print("Initializing components...")
    
    # Create components
    data_source = ArXivDataSource()
    embedding_model = OllamaEmbedding()
    vector_store = ChromaVectorStore(embedding_model)
    
    # Create recommender
    recommender = GaussianRegressionRecommender(data_source, vector_store, embedding_model)
    
    print("Bootstrapping model...")
    success = recommender.bootstrap(force=True)
    
    if success:
        print("\nModel successfully bootstrapped and saved.")
    else:
        print("\nFailed to bootstrap model. Please onboard more papers first.")


def recommend_papers():
    """
    Run a simple CLI interface for paper recommendations.
    """
    from .data_sources import ArXivDataSource
    from .embeddings import OllamaEmbedding
    from .vector_store import ChromaVectorStore
    
    print("\n===== PAPER RECOMMENDATION SYSTEM =====\n")
    
    # Get period from user
    try:
        period = int(input("Enter time period in hours for paper retrieval [default: 48]: ") or "48")
    except ValueError:
        period = 48
        print("Invalid input. Using default: 48 hours")
    
    # Get number of recommendations
    try:
        num_recommendations = int(input("Enter number of recommendations to show [default: 5]: ") or "5")
    except ValueError:
        num_recommendations = 5
        print("Invalid input. Using default: 5 recommendations")
    
    print("\nInitializing recommendation system...")
    
    # Create components
    data_source = ArXivDataSource(period=period)
    embedding_model = OllamaEmbedding()
    vector_store = ChromaVectorStore(embedding_model)
    
    # Create recommender
    recommender = create_recommender(data_source, vector_store, embedding_model)
    
    print("Generating recommendations...")
    recommendations = recommender.recommend(num_recommendations=num_recommendations)
    
    if not recommendations:
        print("\nNo recommendations available. Please onboard more papers first.")
        return
    
    print(f"\n----- TOP {len(recommendations)} RECOMMENDATIONS -----\n")
    
    for i, rec in enumerate(recommendations):
        print(f"Recommendation {i+1}:")
        print(f"Title: {rec['title']}")
        print(f"Link: {rec['link']}")
        print(f"Predicted Rating: {rec['predicted_rating']:.2f}/5.0")
        print(f"Confidence: {rec['confidence']:.2f}")
        
        # Print a preview of the abstract (first 150 chars)
        abstract_preview = rec['abstract'][:150] + "..." if len(rec['abstract']) > 150 else rec['abstract']
        print(f"Abstract preview: {abstract_preview}")
        
        # Ask if user wants to view full abstract
        view_full = input("\nView full abstract? (y/n): ")
        if view_full.lower() == 'y':
            print("\nFull Abstract:")
            print(rec['abstract'])
            input("\nPress Enter to continue...")
        
        print("\n" + "-" * 50 + "\n")
