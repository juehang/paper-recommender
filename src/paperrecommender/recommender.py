import os
import pickle
import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
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
    def __init__(self, data_source, vector_store, embedding_model, max_samples=50, model_path=None, bootstrap_sample_size=20):
        super().__init__(data_source, vector_store, embedding_model)
        self.max_samples = max_samples
        self.model = None
        self.model_path = model_path or os.path.expanduser("~/.paper_recommender/gp_model.pkl")
        self.bootstrap_sample_size = bootstrap_sample_size
        
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
        The model will learn to predict the variance between ratings as a function of similarity.
        
        Returns:
            tuple: (X, y) where X is similarity scores and y is rating variances
        """
        # Get all documents from the vector store
        all_docs = self.vector_store.get_all_documents()
        
        # If we don't have enough data, we can't fit the model
        if len(all_docs["ids"]) < 5:  # Arbitrary minimum threshold
            return None, None
            
        # We'll collect similarity scores and rating variances
        X = []  # Similarity scores
        y = []  # Rating variances (squared differences)
        
        # Sample a subset of documents to use as queries
        sample_size = min(self.bootstrap_sample_size, len(all_docs["ids"]))
        sample_indices = np.random.choice(len(all_docs["ids"]), sample_size, replace=False)
        
        for idx in sample_indices:
            # Get the query document and its rating
            query_text = all_docs["documents"][idx]
            query_rating = float(all_docs["metadatas"][idx].get("rating", 0))
            
            # Search for similar documents
            results = self.vector_store.search([query_text], num_results=self.max_samples)
            
            if "distances" in results and results["distances"]:
                # For each similar document
                for i, (distance, metadata) in enumerate(zip(results["distances"][0], results["metadatas"][0])):
                    similarity = 1 - distance
                    other_rating = float(metadata.get("rating", 0))
                    
                    # Calculate squared difference between ratings
                    rating_variance = (query_rating - other_rating) ** 2
                    
                    # Add to training data
                    X.append([similarity])  # Feature: similarity
                    y.append(rating_variance)  # Target: variance between ratings
        
        # If we don't have enough data points, we can't fit the model
        if len(X) < 10:  # Arbitrary minimum threshold
            return None, None
            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        return X, np.log(y + 0.1)
        
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
        # Using RBF kernel for smoothness + WhiteKernel for observation noise
        kernel = RBF(length_scale=1.0, length_scale_bounds=(5e-3, 1e1)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 20.))
        
        # Create and fit the model
        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
        self.model.fit(X, y)
        
        # Save the model
        self.save_model()
        
        return True
    
    def _gaussian_posterior(self, mean1, var1, mean2, var2):
        return (mean1/var1 + mean2/var2)/(1/var1 + 1/var2), var1*var2/(var1+var2)
    
    def predict_rating_with_sampling(self, document, n_nearest=10, num_samples=100, num_samples_per_posterior=5, N_sigma=1.,):
        """
        Predict rating using GP sampling for robust uncertainty estimation.
        
        Args:
            document (str): Document to predict rating for
            n_nearest (int): Number of nearest embeddings to use
            num_samples (int): Number of GP samples to draw
            num_samples_per_posterior (int): Number of times to sample from each gaussian posterior
            N_sigma (float): Number of standard deviations to use for confidence interval
            
        Returns:
            tuple: (predicted_rating, lower_bound, upper_bound)
        """
        # Check if we have a progress tracker from the embedding model
        progress_tracker = self.embedding_model.progress_tracker
        
        # Get similar documents
        if progress_tracker:
            # Store the original description to restore later
            original_description = progress_tracker.description
            # Only update the description, not the progress bar
            progress_tracker.description = "Finding similar documents"
        
        results = self.vector_store.search([document], num_results=n_nearest)
        
        if "distances" not in results or not results["distances"]:
            return None, None, None
        
        # Get similarities and ratings
        similarities = np.array([1 - dist for dist in results["distances"][0]])
        ratings = np.array([float(meta.get("rating", 0)) for meta in results["metadatas"][0]])

        mask = np.logical_and(ratings<=5, ratings>=1)
        similarities = similarities[mask]
        ratings = ratings[mask]
        
        # Reshape for prediction
        X_pred = similarities.reshape(-1, 1)
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Sampling from Gaussian Process model"
        
        # Draw samples from the GP posterior
        y_samples = np.clip(np.exp(self.model.sample_y(X_pred, num_samples))-0.1, a_min=0.1, a_max=np.inf)
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Calculating rating predictions"
        
        # Calculate statistics from samples
        sample_ratings = []
        for sample_idx in range(num_samples):
            # Get variance predictions for this sample
            sample_variances = y_samples[:, sample_idx]
            
            posterior_mean = ratings[0]
            posterior_var = sample_variances[0]

            for j in range(1, len(ratings)):
                posterior_mean, posterior_var = self._gaussian_posterior(
                    posterior_mean, posterior_var, ratings[j], sample_variances[j]
                )
            
            # Calculate weighted rating for this sample
            sample_ratings.extend(np.random.normal(
                loc=posterior_mean, scale=np.sqrt(posterior_var), size=num_samples_per_posterior
                ))
        
        # Calculate bounds from samples using N_sigma
        # Convert N_sigma to quantiles using the normal distribution CDF
        lower_quantile = stats.norm.cdf(-N_sigma)
        upper_quantile = stats.norm.cdf(N_sigma)
        
        # Get bounds using calculated percentiles
        lower_bound = np.quantile(sample_ratings, lower_quantile)
        upper_bound = np.quantile(sample_ratings, upper_quantile)
        
        # Restore original description
        if progress_tracker and original_description:
            progress_tracker.description = original_description
        
        return np.median(sample_ratings), lower_bound, upper_bound
    
    def recommend(self, num_recommendations=10, exploration_weight=1.0, n_nearest_embeddings=None, gp_num_samples=None, refresh_data=False):
        """
        Recommend new papers from the data source.
        
        Args:
            num_recommendations (int): Number of recommendations to return
            exploration_weight (float): Weight for the exploration term in UCB acquisition.
                0.0 means pure exploitation (use predicted rating only)
                Higher values encourage exploration of uncertain predictions
                Default is 1.0 for balanced exploration-exploitation
            n_nearest_embeddings (int): Number of nearest embeddings to use for prediction
            gp_num_samples (int): Number of GP samples to draw for uncertainty estimation
                
        Returns:
            list: A list of recommended papers with predicted ratings
        """
        # Get a reference to the progress tracker from the embedding model
        from .common import ProgressTracker
        progress_tracker = self.embedding_model.progress_tracker
        
        # If we have a progress tracker, update it to show we're starting
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Preparing recommendation process"
        
        # Ensure we have fresh data
        if refresh_data:
            self.data_source.refresh_data()
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Checking recommendation model"
        
        # Check if we have a model
        if self.model is None:
            # Try to bootstrap if we don't have a model
            success = self.bootstrap()
            if not success:
                # If we couldn't bootstrap, return an empty list
                return []
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Retrieving papers from data source"
        
        # Get new papers from the data source
        titles = self.data_source.titles
        abstracts = self.data_source.abstracts
        links = self.data_source.links
        
        if not titles:
            return []
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Filtering existing papers"
        
        # First pass: count papers that need embeddings (not in vector store)
        papers_to_process = []
        filtered_count = 0
        for title, abstract, link in zip(titles, abstracts, links):
            document = construct_string(title, abstract)
            
            # Check if the document already exists in the vector store
            if self.vector_store.document_exists(document):
                filtered_count += 1
                continue  # Skip this paper
                
            papers_to_process.append((title, abstract, link, document))
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} papers that already exist in the vector store.")
        
        if progress_tracker:
            # Reset the progress tracker specifically for embedding generation
            # This is the only step that will update the progress bar
            progress_tracker.reset(
                total=len(papers_to_process),
                description="Generating embeddings for recommendations"
            )
        
        # Second pass: generate embeddings with progress tracking
        # The embedding_model.get_embedding method will update the progress tracker
        new_papers = []
        for title, abstract, link, document in papers_to_process:
            embedding = self.embedding_model.get_embedding(document)
            
            new_papers.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "document": document,
                "embedding": embedding
            })
            
            # Progress is updated by the embedding_model.get_embedding method
        
        # Use config values if not provided
        from .config import load_config
        config = load_config()
        n_nearest = n_nearest_embeddings or config.get("n_nearest_embeddings", 10)
        num_samples = gp_num_samples or config.get("gp_num_samples", 100)
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Calculating recommendations"
        
        # For each new paper, find similar papers in the vector store
        recommendations = []
        
        for paper in new_papers:
            # Predict rating with sampling, passing exploration_weight as N_sigma
            mean_rating, lower_bound, upper_bound = self.predict_rating_with_sampling(
                paper["document"], n_nearest=n_nearest, num_samples=num_samples, N_sigma=exploration_weight
            )
            
            if mean_rating is None:
                continue
            
            # Calculate uncertainty range
            uncertainty_range = upper_bound - lower_bound
            
            # Calculate acquisition function value
            # Since exploration_weight is now directly controlling the confidence interval width,
            # we can use the upper bound as the acquisition value
            acquisition_value = upper_bound
            
            # Add to recommendations
            recommendations.append({
                "title": paper["title"],
                "abstract": paper["abstract"],
                "link": paper["link"],
                "document": paper["document"],
                "predicted_rating": float(mean_rating),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "uncertainty_range": float(uncertainty_range),
                "acquisition_value": float(acquisition_value)
            })
        
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Sorting and finalizing recommendations"
        
        # Sort by acquisition value (descending) instead of just predicted rating
        recommendations.sort(key=lambda x: x["acquisition_value"], reverse=True)
        
        # Complete the progress
        if progress_tracker:
            # Only update the description, not the progress bar
            progress_tracker.description = "Recommendation process complete"
        
        # Return top N recommendations
        return recommendations[:num_recommendations]


def create_recommender(data_source, vector_store, embedding_model, max_samples=50, model_path=None, bootstrap_sample_size=20):
    """
    Create and initialize a recommender.
    
    Args:
        data_source: The data source to use for new papers
        vector_store: The vector store containing user-rated papers
        embedding_model: The embedding model to use for new papers
        max_samples: Maximum number of samples to use for fitting the model
        model_path: Path to the model pickle file
        bootstrap_sample_size: Number of datapoints to use for GP bootstrap
        
    Returns:
        GaussianRegressionRecommender: An initialized recommender
    """
    recommender = GaussianRegressionRecommender(
        data_source, vector_store, embedding_model, max_samples, model_path, bootstrap_sample_size
    )
    return recommender


def bootstrap_recommender(config=None):
    """
    Run a simple CLI interface for bootstrapping the recommender.
    
    Args:
        config (dict, optional): Configuration dictionary
    """
    from .data_sources import ArXivDataSource
    from .embeddings import create_embedding_model
    from .vector_store import ChromaVectorStore
    from .config import load_config
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    print("\n===== BOOTSTRAPPING RECOMMENDATION SYSTEM =====\n")
    
    print("Initializing components...")
    
    # Create components
    data_source = ArXivDataSource(period=config["period_hours"])
    embedding_model = create_embedding_model(config)
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Create recommender
    recommender = GaussianRegressionRecommender(
        data_source, 
        vector_store, 
        embedding_model,
        max_samples=config["max_samples"],
        model_path=config["model_path"],
        bootstrap_sample_size=config["gp_bootstrap_num_datapoints"]
    )
    
    print("Bootstrapping model...")
    success = recommender.bootstrap(force=True)
    
    if success:
        print("\nModel successfully bootstrapped and saved.")
    else:
        print("\nFailed to bootstrap model. Please onboard more papers first.")


def recommend_papers(config=None):
    """
    Run a simple CLI interface for paper recommendations.
    
    Args:
        config (dict, optional): Configuration dictionary
    """
    from .data_sources import ArXivDataSource
    from .embeddings import create_embedding_model
    from .vector_store import ChromaVectorStore
    from .config import load_config
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    print("\n===== PAPER RECOMMENDATION SYSTEM =====\n")
    
    # Get period from user or use config
    try:
        period_prompt = f"Enter time period in hours for paper retrieval [default: {config['period_hours']}]: "
        period_input = input(period_prompt)
        period = int(period_input) if period_input else config["period_hours"]
    except ValueError:
        period = config["period_hours"]
        print(f"Invalid input. Using default: {period} hours")
    
    # Get number of recommendations
    try:
        num_prompt = f"Enter number of recommendations to show [default: {config['num_recommendations']}]: "
        num_input = input(num_prompt)
        num_recommendations = int(num_input) if num_input else config["num_recommendations"]
    except ValueError:
        num_recommendations = config["num_recommendations"]
        print(f"Invalid input. Using default: {num_recommendations} recommendations")
    
    # Get exploration weight
    try:
        exp_prompt = f"Enter exploration weight (0.0 for pure exploitation) [default: {config['exploration_weight']}]: "
        exp_input = input(exp_prompt)
        exploration_weight = float(exp_input) if exp_input else config["exploration_weight"]
    except ValueError:
        exploration_weight = config["exploration_weight"]
        print(f"Invalid input. Using default: {exploration_weight} (balanced exploration-exploitation)")
    
    # Get n_nearest_embeddings
    try:
        nearest_prompt = f"Enter number of nearest embeddings to use [default: {config['n_nearest_embeddings']}]: "
        nearest_input = input(nearest_prompt)
        n_nearest = int(nearest_input) if nearest_input else config["n_nearest_embeddings"]
    except ValueError:
        n_nearest = config["n_nearest_embeddings"]
        print(f"Invalid input. Using default: {n_nearest} nearest embeddings")
    
    # Get gp_num_samples
    try:
        samples_prompt = f"Enter number of GP samples [default: {config['gp_num_samples']}]: "
        samples_input = input(samples_prompt)
        gp_samples = int(samples_input) if samples_input else config["gp_num_samples"]
    except ValueError:
        gp_samples = config["gp_num_samples"]
        print(f"Invalid input. Using default: {gp_samples} GP samples")
    
    print("\nInitializing recommendation system...")
    
    # Create components
    data_source = ArXivDataSource(period=period)
    embedding_model = create_embedding_model(config)
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Create recommender
    recommender = create_recommender(
        data_source, 
        vector_store, 
        embedding_model,
        max_samples=config["max_samples"],
        model_path=config["model_path"],
        bootstrap_sample_size=config["gp_bootstrap_num_datapoints"]
    )
    
    print("Generating recommendations...")
    recommendations = recommender.recommend(
        num_recommendations=num_recommendations,
        exploration_weight=exploration_weight,
        n_nearest_embeddings=n_nearest,
        gp_num_samples=gp_samples
    )
    
    if not recommendations:
        print("\nNo recommendations available. Please onboard more papers first.")
        return
    
    print(f"\n----- TOP {len(recommendations)} RECOMMENDATIONS -----\n")
    
    # Track which papers were rated
    rated_papers = []
    
    for i, rec in enumerate(recommendations):
        print(f"Recommendation {i+1}:")
        print(f"Title: {rec['title']}")
        print(f"Link: {rec['link']}")
        
        # Check if we have the new prediction format
        if 'lower_bound' in rec and 'upper_bound' in rec:
            print(f"Predicted Rating: {rec['predicted_rating']:.2f}/5.0")
            print(f"Rating Range: [{rec['lower_bound']:.2f}, {rec['upper_bound']:.2f}]")
            print(f"Uncertainty Range: {rec['uncertainty_range']:.4f}")
        else:
            # Fallback to old format
            print(f"Predicted Rating: {rec['predicted_rating']:.2f}/5.0")
            print(f"Uncertainty (std dev): {rec['std_dev']:.4f}")
            
        print(f"Acquisition Value: {rec['acquisition_value']:.4f}")
        
        # Print a preview of the abstract (first 150 chars)
        abstract_preview = rec['abstract'][:150] + "..." if len(rec['abstract']) > 150 else rec['abstract']
        print(f"Abstract preview: {abstract_preview}")
        
        # Ask if user wants to view full abstract
        view_full = input("\nView full abstract? (y/n): ")
        if view_full.lower() == 'y':
            print("\nFull Abstract:")
            print(rec['abstract'])
            input("\nPress Enter to continue...")
        
        # Ask if user wants to rate this paper
        rate_paper = input("\nWould you like to rate this paper? (y/n): ")
        if rate_paper.lower() == 'y':
            while True:
                try:
                    rating = int(input("Enter rating (1-5): "))
                    if 1 <= rating <= 5:
                        # Add to list of rated papers
                        rated_papers.append({
                            "document": rec["document"],
                            "link": rec["link"],
                            "rating": rating
                        })
                        print(f"Paper rated: {rating}/5")
                        break
                    else:
                        print("Please enter a rating between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5.")
        
        print("\n" + "-" * 50 + "\n")
    
    # If any papers were rated, ask if user wants to add them to the vector store
    if rated_papers:
        add_to_store = input("\nWould you like to add the rated papers to your vector store? (y/n): ")
        if add_to_store.lower() == 'y':
            # Add papers to vector store
            for paper in rated_papers:
                vector_store.add_document(
                    paper["document"],
                    paper["link"],
                    paper["rating"]
                )
            
            print(f"\nAdded {len(rated_papers)} papers to the vector store.")
            
            # Ask if user wants to bootstrap the recommender
            bootstrap_model = input("Would you like to update the recommendation model with the new data? (y/n): ")
            if bootstrap_model.lower() == 'y':
                recommender.bootstrap(force=True)
                print("Recommendation model updated.")
