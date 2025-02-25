#!/usr/bin/env python3
import argparse
import os
from .config import load_config, update_config_from_args, is_first_startup
from .onboarding import terminal_ui_onboarding, create_onboarding_system
from .recommender import recommend_papers, bootstrap_recommender, create_recommender
from .data_sources import ArXivDataSource
from .embeddings import OllamaEmbedding
from .vector_store import ChromaVectorStore

def main():
    """
    Main entry point for the paper recommender CLI.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Paper Recommender")
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--onboard", action="store_true", help="Run onboarding process")
    group.add_argument("--recommend", action="store_true", help="Run recommendation process")
    group.add_argument("--bootstrap", action="store_true", help="Bootstrap the recommendation model")
    
    # Configuration
    parser.add_argument("--config", help="Path to custom config file")
    
    # Override configuration parameters
    parser.add_argument("--chroma-db-path", help="Path to ChromaDB directory")
    parser.add_argument("--model-path", help="Path to model pickle file")
    parser.add_argument("--exploration-weight", type=float, help="Exploration weight for recommendations")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples for similarity search")
    parser.add_argument("--period-hours", type=int, help="Time period in hours for paper retrieval")
    parser.add_argument("--random-sample-size", type=int, help="Number of random papers to select during onboarding")
    parser.add_argument("--diverse-sample-size", type=int, help="Number of diverse papers to select during onboarding")
    parser.add_argument("--num-recommendations", type=int, help="Number of recommendations to show")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command-line arguments
    config = update_config_from_args(config, args)
    
    # Create components with configuration
    data_source = ArXivDataSource(period=config["period_hours"])
    embedding_model = OllamaEmbedding()
    vector_store = ChromaVectorStore(embedding_model, path=config["chroma_db_path"])
    
    # Determine what to run
    first_startup = is_first_startup()
    
    if args.bootstrap:
        print("\n===== BOOTSTRAPPING RECOMMENDATION MODEL =====\n")
        # Use the bootstrap_recommender function with configuration
        bootstrap_recommender(config)
    elif args.onboard:
        # Run onboarding process with configuration
        _, embedding_model, vector_store, onboarder = create_onboarding_system(
            period=config["period_hours"],
            random_sample_size=config["random_sample_size"],
            diverse_sample_size=config["diverse_sample_size"],
            chroma_db_path=config["chroma_db_path"]
        )
        terminal_ui_onboarding(config)
    elif args.recommend:
        # Run recommendation process with configuration
        recommender = create_recommender(
            data_source, 
            vector_store, 
            embedding_model,
            max_samples=config["max_samples"],
            model_path=config["model_path"]
        )
        recommend_papers(config)
    else:
        # Default behavior based on first startup
        if first_startup:
            print("First startup detected. Running onboarding process...")
            _, embedding_model, vector_store, onboarder = create_onboarding_system(
                period=config["period_hours"],
                random_sample_size=config["random_sample_size"],
                diverse_sample_size=config["diverse_sample_size"],
                chroma_db_path=config["chroma_db_path"]
            )
            terminal_ui_onboarding(config)
        else:
            print("Running recommendation process...")
            recommender = create_recommender(
                data_source, 
                vector_store, 
                embedding_model,
                max_samples=config["max_samples"],
                model_path=config["model_path"]
            )
            recommend_papers(config)

if __name__ == "__main__":
    main()
