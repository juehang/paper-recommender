import os
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "chroma_db_path": os.path.expanduser("~/.paper_recommender/chroma_db"),
    "model_path": os.path.expanduser("~/.paper_recommender/gp_model.pkl"),
    "embedding_cache_path": os.path.expanduser("~/.paper_recommender/embedding_cache.pkl"),
    "embedding_provider": "ollama",  # Embedding provider to use ("ollama" or "openai")
    "openai_api_key": "",  # OpenAI API key (only used when embedding_provider is "openai")
    "openai_embedding_model": "text-embedding-ada-002",  # OpenAI embedding model to use
    "exploration_weight": 0.5,
    "max_samples": 20,
    "period_hours": 48,
    "random_sample_size": 5,
    "diverse_sample_size": 5,
    "num_recommendations": 5,
    "gp_num_samples": 100,  # Number of GP samples for uncertainty estimation
    "gp_bootstrap_num_datapoints": 200,  # Number of datapoints to use for GP bootstrap
    "n_nearest_embeddings": 10  # Number of nearest embeddings to use for prediction
}

# Configuration file path
CONFIG_DIR = os.path.expanduser("~/.paper_recommender")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def ensure_config_dir():
    """Ensure the configuration directory exists."""
    os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config(custom_config_path=None):
    """
    Load configuration from file or create default if it doesn't exist.
    
    Args:
        custom_config_path (str, optional): Path to a custom config file
        
    Returns:
        dict: Configuration dictionary
    """
    ensure_config_dir()
    
    config_path = custom_config_path or CONFIG_FILE
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Update with any missing default values
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration.")
            return DEFAULT_CONFIG.copy()
    else:
        # Create default config file if it's the standard location
        if config_path == CONFIG_FILE:
            save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config, custom_config_path=None):
    """
    Save configuration to file.
    
    Args:
        config (dict): Configuration dictionary
        custom_config_path (str, optional): Path to a custom config file
    """
    ensure_config_dir()
    
    config_path = custom_config_path or CONFIG_FILE
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving config file: {e}")

def is_first_startup():
    """
    Check if this is the first startup by checking if the ChromaDB directory exists.
    
    Returns:
        bool: True if this is the first startup, False otherwise
    """
    config = load_config()
    chroma_path = Path(config["chroma_db_path"])
    return not chroma_path.exists()

def update_config_from_args(config, args):
    """
    Update configuration with command-line arguments.
    
    Args:
        config (dict): Configuration dictionary
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        dict: Updated configuration dictionary
    """
    # Convert args to dictionary, filtering out None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    # Map argument names to config keys
    arg_to_config = {
        "chroma_db_path": "chroma_db_path",
        "model_path": "model_path",
        "embedding_cache_path": "embedding_cache_path",
        "embedding_provider": "embedding_provider",
        "openai_api_key": "openai_api_key",
        "openai_embedding_model": "openai_embedding_model",
        "exploration_weight": "exploration_weight",
        "max_samples": "max_samples",
        "period_hours": "period_hours",
        "random_sample_size": "random_sample_size",
        "diverse_sample_size": "diverse_sample_size",
        "num_recommendations": "num_recommendations",
        "gp_num_samples": "gp_num_samples",
        "gp_bootstrap_num_datapoints": "gp_bootstrap_num_datapoints",
        "n_nearest_embeddings": "n_nearest_embeddings"
    }
    
    # Update config with args
    for arg_name, config_key in arg_to_config.items():
        if arg_name in args_dict:
            config[config_key] = args_dict[arg_name]
    
    return config
