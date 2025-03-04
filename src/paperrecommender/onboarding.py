# Structure for the onboarding module

from .common import construct_string, cosine_similarity
from .data_sources import ArXivDataSource
from .embeddings import create_embedding_model
from .vector_store import ChromaVectorStore
import random
import numpy as np

class OnboardingStrategy:
    """
    Base strategy class for selecting papers during onboarding.
    """
    def __init__(self, data_source, embedding_model, sample_size=5):
        self.data_source = data_source
        self.embedding_model = embedding_model
        self.sample_size = sample_size
        
    def select_papers(self, used_indices=None, onboarded_embeddings=None):
        """
        Select papers based on the strategy.
        
        Args:
            used_indices (set): Indices of papers that have already been selected
            onboarded_embeddings (list): Embeddings of papers already onboarded
            
        Returns:
            tuple: (titles, abstracts, links, used_indices)
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def estimate_papers_to_process(self, used_indices=None):
        """
        Estimate how many papers this strategy will need to process.
        
        Args:
            used_indices (set): Indices of papers that have already been selected
            
        Returns:
            int: Estimated number of papers to process
        """
        raise NotImplementedError("Subclasses must implement this method")


class RandomSelectionStrategy(OnboardingStrategy):
    """
    Strategy to randomly select papers.
    """
    def estimate_papers_to_process(self, used_indices=None):
        """
        For random selection, we only process the sample size papers.
        
        Args:
            used_indices (set): Indices of papers that have already been selected
            
        Returns:
            int: Estimated number of papers to process
        """
        if used_indices is None:
            used_indices = set()
            
        # Get all available data
        available_count = len(self.data_source.titles)
        available_indices = [i for i in range(available_count) if i not in used_indices]
        
        # We'll process at most sample_size papers
        return min(self.sample_size, len(available_indices))
    
    def select_papers(self, used_indices=None, onboarded_embeddings=None):
        if used_indices is None:
            used_indices = set()
            
        # Get all available data
        titles = self.data_source.titles
        abstracts = self.data_source.abstracts
        links = self.data_source.links
        
        # Check if we have enough papers
        available_count = len(titles)
        available_indices = [i for i in range(available_count) if i not in used_indices]
        actual_sample_size = min(self.sample_size, len(available_indices))
        
        if actual_sample_size == 0:
            return [], [], [], used_indices
            
        # Select random indices without replacement
        selected_indices = random.sample(available_indices, actual_sample_size)
        
        # Create the sample
        sampled_titles = [titles[i] for i in selected_indices]
        sampled_abstracts = [abstracts[i] for i in selected_indices]
        sampled_links = [links[i] for i in selected_indices]
        
        # Update used indices
        used_indices.update(selected_indices)
        
        return sampled_titles, sampled_abstracts, sampled_links, used_indices


class DiverseSelectionStrategy(OnboardingStrategy):
    """
    Strategy to select papers that are most diverse from existing ones.
    """
    def estimate_papers_to_process(self, used_indices=None):
        """
        For diverse selection, we need to process all remaining papers.
        
        Args:
            used_indices (set): Indices of papers that have already been selected
            
        Returns:
            int: Estimated number of papers to process
        """
        if used_indices is None:
            used_indices = set()
            
        # Get all available data
        available_count = len(self.data_source.titles)
        
        # We need to process all papers that haven't been used yet
        return available_count - len(used_indices)
    
    def select_papers(self, used_indices=None, onboarded_embeddings=None):
        if used_indices is None:
            used_indices = set()
        if onboarded_embeddings is None:
            onboarded_embeddings = []
            
        titles = self.data_source.titles
        abstracts = self.data_source.abstracts
        links = self.data_source.links
        
        available_count = len(titles)
        # Skip if we don't have any papers or all papers are already used
        if available_count == 0 or len(used_indices) >= available_count:
            return [], [], [], used_indices
            
        # Calculate embeddings for all remaining papers
        remaining_indices = [i for i in range(available_count) if i not in used_indices]
        remaining_embeddings = []
        
        # The embedding model will automatically handle progress tracking and caching
        # We don't need to reset the total here as the embedding model will count
        # only uncached items when calculating progress
        progress_tracker_set = False
        if hasattr(self.embedding_model, 'progress_tracker') and self.embedding_model.progress_tracker:
            self.embedding_model.progress_tracker.reset(
                description="Generating embeddings for diverse selection"
            )
            progress_tracker_set = True
        
        try:
            for i in remaining_indices:
                text = construct_string(titles[i], abstracts[i])
                embedding = self.embedding_model.get_embedding(text)
                remaining_embeddings.append((i, embedding))
        finally:
            # Close the progress tracker if we set it
            if progress_tracker_set and hasattr(self.embedding_model, 'progress_tracker') and self.embedding_model.progress_tracker:
                self.embedding_model.progress_tracker.close()
        
        # If we don't have any onboarded embeddings yet, just pick randomly
        if not onboarded_embeddings:
            random_indices = random.sample(remaining_indices, 
                                min(self.sample_size, len(remaining_indices)))
            selected_indices = random_indices
        else:
            # Calculate the "diversity score" for each remaining paper
            diversity_scores = []
            
            for i, embed in remaining_embeddings:
                # For each remaining paper, find its maximum similarity to any onboarded paper
                max_similarity = -1  # Cosine similarity ranges from -1 to 1
                
                for onboarded_embed in onboarded_embeddings:
                    try:
                        similarity = cosine_similarity(embed, onboarded_embed)
                        max_similarity = max(max_similarity, similarity)
                    except ValueError:
                        # Handle possible zero vector error
                        continue
                
                # Lower similarity means more diverse
                diversity_score = 1 - max_similarity if max_similarity != -1 else 2  # Prioritize if no valid comparison
                diversity_scores.append((i, diversity_score))
            
            # Sort by diversity score (higher is more diverse)
            diversity_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top most diverse papers
            selected_indices = [pair[0] for pair in diversity_scores[:self.sample_size]]
        
        # Extract the selected diverse papers
        selected_titles = [titles[i] for i in selected_indices]
        selected_abstracts = [abstracts[i] for i in selected_indices]
        selected_links = [links[i] for i in selected_indices]
        
        # Update used indices
        used_indices.update(selected_indices)
        
        return selected_titles, selected_abstracts, selected_links, used_indices


class Onboarder:
    """
    Class to handle the onboarding process using different selection strategies,
    with support for web UI integration.
    """
    def __init__(self, data_source, vector_store, embedding_model, strategies=None):
        """
        Initialize the onboarder with a data source, vector store, and strategies.
        
        Args:
            data_source: The data source object (e.g., ArXivDataSource)
            vector_store: The vector store object (e.g., ChromaVectorStore)
            embedding_model: The embedding model for creating vectors
            strategies (list): List of selection strategies to use
        """
        self.data_source = data_source
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        # Default strategies if none provided
        if strategies is None:
            strategies = [
                RandomSelectionStrategy(data_source, embedding_model),
                DiverseSelectionStrategy(data_source, embedding_model)
            ]
        self.strategies = strategies
        self.selected_papers = []
        self.onboarded_embeddings = []
        self.used_indices = set()
        
    def prepare_candidates(self):
        """
        Prepare candidate papers for onboarding without actually storing them.
        This allows a UI to display the papers and let users rate them.
        
        Returns:
            list: A list of dictionaries containing paper details for each strategy
        """
        # Ensure data is fresh
        self.data_source.refresh_data()
        
        self.used_indices = set()
        self.onboarded_embeddings = []
        self.selected_papers = []
        
        # Create a progress tracker for the embedding process if not already set
        if not hasattr(self.embedding_model, 'progress_tracker') or not self.embedding_model.progress_tracker:
            # Get a strategy-dependent estimate of papers to process
            # This will be more accurate than just using sample_size
            total_papers = sum(strategy.estimate_papers_to_process(self.used_indices) 
                              for strategy in self.strategies)
            
            from .common import ProgressTracker
            progress_tracker = ProgressTracker(
                total=total_papers, 
                description="Generating embeddings"
            )
            self.embedding_model.set_progress_tracker(progress_tracker)
        
        strategy_papers = []
        
        try:
            # Apply each strategy in sequence
            for i, strategy in enumerate(self.strategies):
                titles, abstracts, links, self.used_indices = strategy.select_papers(
                    self.used_indices, self.onboarded_embeddings
                )
                
                # Filter out papers that already exist in the vector store
                filtered_titles = []
                filtered_abstracts = []
                filtered_links = []
                filtered_count = 0
                
                for title, abstract, link in zip(titles, abstracts, links):
                    document = construct_string(title, abstract)
                    if not self.vector_store.document_exists(document):
                        filtered_titles.append(title)
                        filtered_abstracts.append(abstract)
                        filtered_links.append(link)
                    else:
                        filtered_count += 1
                
                if filtered_count > 0:
                    print(f"Filtered out {filtered_count} papers that already exist in the vector store.")
                
                strategy_result = []
                for title, abstract, link in zip(filtered_titles, filtered_abstracts, filtered_links):
                    document = construct_string(title, abstract)
                    embedding = self.embedding_model.get_embedding(document)
                    
                    paper = {
                        'title': title,
                        'abstract': abstract,
                        'link': link,
                        'document': document,
                        'embedding': embedding,
                        'strategy_index': i,
                        'rating': None  # To be filled by the UI
                    }
                    strategy_result.append(paper)
                    self.selected_papers.append(paper)
                    self.onboarded_embeddings.append(embedding)
                    
                strategy_papers.append(strategy_result)
        finally:
            # Ensure progress tracker is closed even if an exception occurs
            if hasattr(self.embedding_model, 'progress_tracker') and self.embedding_model.progress_tracker:
                self.embedding_model.progress_tracker.close()
                # Reset the progress tracker in the embedding model
                self.embedding_model.set_progress_tracker(None)
            
        return strategy_papers
    
    def get_strategy_names(self):
        """
        Get the names of the strategies being used.
        
        Returns:
            list: Names of the strategies (using class names)
        """
        return [strategy.__class__.__name__ for strategy in self.strategies]
    
    def get_candidate_count(self):
        """
        Get the number of papers selected by each strategy.
        
        Returns:
            list: Number of papers from each strategy
        """
        return [len(papers) for papers in self.prepare_candidates()]
    
    def set_paper_rating(self, paper_index, rating):
        """
        Set the rating for a paper by its index in the selected_papers list.
        
        Args:
            paper_index (int): Index of the paper in the selected_papers list
            rating (int/float): Rating value to assign
            
        Returns:
            bool: True if successful, False otherwise
        """
        if 0 <= paper_index < len(self.selected_papers):
            self.selected_papers[paper_index]['rating'] = rating
            return True
        return False
    
    def set_ratings_batch(self, ratings):
        """
        Set ratings for multiple papers at once.
        
        Args:
            ratings (list): List of (index, rating) tuples
            
        Returns:
            int: Number of ratings successfully set
        """
        success_count = 0
        for idx, rating in ratings:
            if self.set_paper_rating(idx, rating):
                success_count += 1
        return success_count
    
    def commit_onboarding(self, default_rating=0, bootstrap_recommender=True):
        """
        Commit the onboarding process by storing papers with their ratings.
        Papers without ratings will use the default_rating.
        
        Args:
            default_rating: Rating to use for papers without an explicit rating
            bootstrap_recommender: Whether to bootstrap the recommender after onboarding
            
        Returns:
            dict: Statistics about the onboarding process
        """
        strategy_counts = [0] * len(self.strategies)
        total_count = 0
        
        for paper in self.selected_papers:
            # Use the provided rating or default
            rating = paper['rating'] if paper['rating'] is not None else default_rating
            
            # Add to vector store
            self.vector_store.add_document(
                paper['document'], 
                paper['link'], 
                rating
            )
            
            # Update statistics
            strategy_counts[paper['strategy_index']] += 1
            total_count += 1
        
        # Clear the staging area
        self.selected_papers = []
        
        # Bootstrap the recommender if requested
        if bootstrap_recommender and total_count > 0:
            try:
                from .recommender import GaussianRegressionRecommender
                recommender = GaussianRegressionRecommender(None, self.vector_store, self.embedding_model)
                recommender.bootstrap(force=True)
                print("Recommender model bootstrapped with new data.")
            except Exception as e:
                print(f"Failed to bootstrap recommender: {e}")
        
        return {
            'strategy_counts': strategy_counts,
            'total_count': total_count,
            'strategy_names': self.get_strategy_names()
        }
    
    def simple_onboard(self, default_rating=0):
        """
        Simplified onboarding process for cases where UI interaction isn't needed.
        This prepares candidates and immediately commits them with the default rating.
        
        Args:
            default_rating: Rating to use for all papers
            
        Returns:
            dict: Statistics about the onboarding process
        """
        self.prepare_candidates()
        return self.commit_onboarding(default_rating)
        
    def add_custom_paper(self, title, abstract, link, rating):
        """
        Add a custom paper provided by the user.
        
        Args:
            title (str): The title of the paper
            abstract (str): The abstract of the paper
            link (str): The link to the paper
            rating (int): The user's rating of the paper (1-5)
            
        Returns:
            bool: True if the paper was added successfully, False otherwise
        """
        # Construct the document string
        document = construct_string(title, abstract)
        
        # Check if the paper already exists in the vector store
        if self.vector_store.document_exists(document):
            print(f"Paper already exists in the vector store: {title}")
            return False
        
        # Generate embedding for the paper
        try:
            embedding = self.embedding_model.get_embedding(document)
            
            # Add the paper to the vector store
            self.vector_store.add_document(document, link, rating)
        finally:
            # Ensure progress tracker is closed if it exists
            if hasattr(self.embedding_model, 'progress_tracker') and self.embedding_model.progress_tracker:
                self.embedding_model.progress_tracker.close()
                self.embedding_model.set_progress_tracker(None)
        
        print(f"Added custom paper to the vector store: {title}")
        return True


# Factory function with the same interface
def create_onboarding_system(period=48, random_sample_size=5, diverse_sample_size=5, chroma_db_path=None, embedding_cache_path=None, config=None):
    """
    Create and return a complete onboarding system.
    
    Args:
        period (int): Time period in hours for fetching recent papers
        random_sample_size (int): Number of papers for random selection
        diverse_sample_size (int): Number of papers for diverse selection
        chroma_db_path (str, optional): Path to ChromaDB directory
        embedding_cache_path (str, optional): Path to embedding cache file
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: (data_source, embedding_model, vector_store, onboarder)
    """
    from .config import load_config
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Override config with provided parameters
    if period is not None:
        config["period_hours"] = period
    if chroma_db_path is not None:
        config["chroma_db_path"] = chroma_db_path
    if embedding_cache_path is not None:
        config["embedding_cache_path"] = embedding_cache_path
    
    # Create the components
    data_source = ArXivDataSource(period=config["period_hours"])
    embedding_model = create_embedding_model(config)
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Create strategies
    random_strategy = RandomSelectionStrategy(data_source, embedding_model, random_sample_size)
    diverse_strategy = DiverseSelectionStrategy(data_source, embedding_model, diverse_sample_size)
    
    # Create onboarder with strategies
    onboarder = Onboarder(data_source, vector_store, embedding_model, 
                          [random_strategy, diverse_strategy])
    
    return data_source, embedding_model, vector_store, onboarder

def terminal_ui_onboarding(config=None):
    """
    Run a simple terminal-based UI for the onboarding process.
    
    Args:
        config (dict, optional): Configuration dictionary
    """
    from .config import load_config
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    print("\n===== PAPER ONBOARDING SYSTEM =====\n")
    
    # Initialize components with default configuration
    print("Initializing onboarding system...")
    data_source = ArXivDataSource(period=config["period_hours"])
    embedding_model = create_embedding_model(config)
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Create strategies with default sample sizes
    random_strategy = RandomSelectionStrategy(data_source, embedding_model, config["random_sample_size"])
    diverse_strategy = DiverseSelectionStrategy(data_source, embedding_model, config["diverse_sample_size"])
    
    # Create onboarder with strategies
    onboarder = Onboarder(data_source, vector_store, embedding_model, 
                          [random_strategy, diverse_strategy])
    
    # Ask if the user wants to add a custom paper
    add_custom = input("Would you like to add a custom paper? (y/n): ")
    if add_custom.lower() == 'y':
        custom_papers_added = 0
        while True:
            print("\n----- ADD CUSTOM PAPER -----\n")
            
            # Get paper details
            title = input("Enter paper title: ")
            if not title:
                print("Title cannot be empty. Skipping this paper.")
                continue
                
            abstract = input("Enter paper abstract: ")
            if not abstract:
                print("Abstract cannot be empty. Skipping this paper.")
                continue
                
            link = input("Enter paper link (optional): ")
            
            # Get rating
            while True:
                try:
                    rating_input = input("Rate this paper (1-5): ")
                    rating = int(rating_input)
                    if 1 <= rating <= 5:
                        break
                    else:
                        print("Please enter a rating between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5.")
            
            # Add the paper
            success = onboarder.add_custom_paper(title, abstract, link, rating)
            if success:
                custom_papers_added += 1
            
            # Ask if the user wants to add another paper
            add_another = input("\nAdd another custom paper? (y/n): ")
            if add_another.lower() != 'y':
                break
        
        print(f"\nAdded {custom_papers_added} custom papers to the vector store.")
        
        # Ask if the user wants to bootstrap the recommender
        if custom_papers_added > 0:
            bootstrap_model = input("Would you like to update the recommendation model with the new data? (y/n): ")
            if bootstrap_model.lower() == 'y':
                try:
                    from .recommender import GaussianRegressionRecommender
                    recommender = GaussianRegressionRecommender(None, vector_store, embedding_model)
                    recommender.bootstrap(force=True)
                    print("Recommendation model updated.")
                except Exception as e:
                    print(f"Failed to bootstrap recommender: {e}")
        
        # Ask if the user wants to continue with the automated onboarding
        continue_auto = input("\nContinue with automated paper onboarding? (y/n): ")
        if continue_auto.lower() != 'y':
            return
    
    # Get period from user or use config
    try:
        period_prompt = f"Enter time period in hours for paper retrieval [default: {config['period_hours']}]: "
        period_input = input(period_prompt)
        period = int(period_input) if period_input else config["period_hours"]
    except ValueError:
        period = config["period_hours"]
        print(f"Invalid input. Using default: {period} hours")
    
    # Get sample sizes from user or use config
    try:
        random_prompt = f"Enter number of random papers to select [default: {config['random_sample_size']}]: "
        random_input = input(random_prompt)
        random_size = int(random_input) if random_input else config["random_sample_size"]
    except ValueError:
        random_size = config["random_sample_size"]
        print(f"Invalid input. Using default: {random_size} papers")
        
    try:
        diverse_prompt = f"Enter number of diverse papers to select [default: {config['diverse_sample_size']}]: "
        diverse_input = input(diverse_prompt)
        diverse_size = int(diverse_input) if diverse_input else config["diverse_sample_size"]
    except ValueError:
        diverse_size = config["diverse_sample_size"]
        print(f"Invalid input. Using default: {diverse_size} papers")
    
    # If user provided custom values, update the components
    if period != config["period_hours"] or random_size != config["random_sample_size"] or diverse_size != config["diverse_sample_size"]:
        print("\nUpdating onboarding system with custom parameters...")
        
        # Update data source if period changed
        if period != config["period_hours"]:
            data_source = ArXivDataSource(period=period)
            
        # Update strategies with new sample sizes
        random_strategy = RandomSelectionStrategy(data_source, embedding_model, random_size)
        diverse_strategy = DiverseSelectionStrategy(data_source, embedding_model, diverse_size)
        
        # Recreate onboarder with updated strategies
        onboarder = Onboarder(data_source, vector_store, embedding_model, 
                              [random_strategy, diverse_strategy])
    
    # The progress tracker will be created in prepare_candidates with the appropriate total
    print("Preparing candidate papers...")
    strategy_papers = onboarder.prepare_candidates()
    strategy_names = onboarder.get_strategy_names()
    
    # Process each strategy's papers
    for strategy_index, (strategy_name, papers) in enumerate(zip(strategy_names, strategy_papers)):
        print(f"\n----- {strategy_name} -----")
        print(f"Selected {len(papers)} papers\n")
        
        for paper_index, paper in enumerate(papers):
            # Find global index of this paper in the onboarder's selected_papers list
            global_index = onboarder.selected_papers.index(paper)
            
            # Print paper details
            print(f"Paper {paper_index + 1}/{len(papers)}:")
            print(f"Title: {paper['title']}")
            print(f"Link: {paper['link']}")
            
            # Print a preview of the abstract (first 150 chars)
            abstract_preview = paper['abstract'][:150] + "..." if len(paper['abstract']) > 150 else paper['abstract']
            print(f"Abstract preview: {abstract_preview}")
            
            # Ask if user wants to view full abstract
            view_full = input("\nView full abstract? (y/n): ")
            if view_full.lower() == 'y':
                print("\nFull Abstract:")
                print(paper['abstract'])
                input("\nPress Enter to continue...")
            
            # Get user rating
            while True:
                rating_input = input("\nRate this paper (1-5, or 's' to skip): ")
                
                if rating_input.lower() == 's':
                    print("Paper skipped (will use default rating).")
                    break
                
                try:
                    rating = int(rating_input)
                    if 1 <= rating <= 5:
                        onboarder.set_paper_rating(global_index, rating)
                        print(f"Paper rated: {rating}/5")
                        break
                    else:
                        print("Please enter a rating between 1 and 5.")
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 5 or 's' to skip.")
            
            
            print("\n" + "-" * 50 + "\n")
        
        # Ask if user wants to continue to next strategy
        if strategy_index < len(strategy_names) - 1:
            continue_input = input(f"Continue to {strategy_names[strategy_index + 1]} papers? (y/n): ")
            if continue_input.lower() != 'y':
                print("Onboarding process stopped by user.")
                return
    
    # Confirm final onboarding
    print("\n----- ONBOARDING SUMMARY -----")
    rated_count = sum(1 for paper in onboarder.selected_papers if paper['rating'] is not None)
    total_count = len(onboarder.selected_papers)
    
    print(f"You have rated {rated_count} out of {total_count} papers.")
    
    try:
        default_rating = int(input("Enter default rating for skipped papers [default: 0]: ") or "0")
        if not (0 <= default_rating <= 5):
            print("Default rating should be between 0 and 5. Using 0.")
            default_rating = 0
    except ValueError:
        default_rating = 0
        print("Invalid input. Using default rating: 0")
    
    confirm = input("\nCommit all papers to the vector store? (y/n): ")
    if confirm.lower() == 'y':
        results = onboarder.commit_onboarding(default_rating)
        
        print("\n----- ONBOARDING COMPLETE -----")
        for strategy_name, count in zip(results['strategy_names'], results['strategy_counts']):
            print(f"  - {strategy_name}: {count} papers")
        print(f"Total: {results['total_count']} papers added to the vector store")
        
        # Offer a simple search to test the results
        do_search = input("\nWould you like to test a search query? (y/n): ")
        if do_search.lower() == 'y':
            query = input("Enter your search query: ")
            try:
                num_results = int(input("Number of results to display [default: 3]: ") or "3")
            except ValueError:
                num_results = 3
            
            print("\nSearching...")
            results = vector_store.search([query], num_results=num_results)
            
            print("\n----- SEARCH RESULTS -----")
            if 'distances' in results and results['distances']:
                for i, (distance, metadata) in enumerate(zip(results['distances'][0], results['metadatas'][0])):
                    print(f"Result {i+1}:")
                    print(f"  Similarity: {1 - distance:.4f}")  # Convert distance to similarity
                    print(f"  Rating: {metadata.get('rating', 'N/A')}")
                    print(f"  Link: {metadata.get('link', 'N/A')}")  # Fixed: 'link' instead of 'links'
                    print()
            else:
                print("No results found.")
    else:
        print("Onboarding cancelled. No papers were added to the vector store.")
