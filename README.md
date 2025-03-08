# Paper Recommender

A research paper recommendation engine that uses text embeddings and Gaussian Process Regression (GPR) to suggest relevant papers based on your preferences.

## Overview

Paper Recommender helps researchers discover relevant papers by:

1. Fetching recent papers from arXiv
2. Using text embeddings to represent paper content
3. Learning your preferences through an onboarding process
4. Recommending papers using a variance-based Gaussian Process Regression model
5. Balancing exploration and exploitation to improve recommendations over time

The system uses a sophisticated approach where the GPR model learns the expected variance between ratings as a function of similarity, rather than directly predicting ratings. This provides more robust recommendations with better uncertainty estimates.

## Installation

### Prerequisites

- Python 3.8 or higher
- One of the following for text embeddings:
  - [Ollama](https://ollama.ai/) (default)
  - OpenAI API key (optional)

### Installing Paper Recommender

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paper-recommender.git
   cd paper-recommender
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

### Setting up Ollama with nomic-embed-text

The paper recommender uses Ollama with the nomic-embed-text model for generating text embeddings.

1. Install Ollama by following the instructions at [ollama.ai](https://ollama.ai/)

2. Pull the nomic-embed-text model:
   ```bash
   ollama pull nomic-embed-text
   ```

3. Verify the installation:
   ```bash
   ollama run nomic-embed-text "Hello, world!"
   ```

4. Ensure the Ollama server is running when using Paper Recommender:
   ```bash
   ollama serve
   ```

## Configuration

Paper Recommender uses a configuration file located at `~/.paper_recommender/config.json`. The default configuration is created on first run, but you can customize it:

```json
{
  "chroma_db_path": "~/.paper_recommender/chroma_db",
  "model_path": "~/.paper_recommender/gp_model.pkl",
  "embedding_cache_path": "~/.paper_recommender/embedding_cache.pkl",
  "embedding_provider": "ollama",
  "openai_api_key": "",
  "openai_embedding_model": "text-embedding-ada-002",
  "exploration_weight": 1.0,
  "max_samples": 1000,
  "period_hours": 48,
  "random_sample_size": 5,
  "diverse_sample_size": 5,
  "num_recommendations": 5,
  "gp_num_samples": 100,
  "n_nearest_embeddings": 10
}
```

Key configuration parameters:
- `embedding_provider`: Which embedding provider to use ("ollama" or "openai")
- `openai_api_key`: Your OpenAI API key (only used when embedding_provider is "openai")
- `openai_embedding_model`: The OpenAI embedding model to use (default: "text-embedding-ada-002")
- `exploration_weight`: Controls the balance between exploration and exploitation (higher values favor exploration)
- `period_hours`: Time window for fetching recent papers from arXiv
- `random_sample_size` and `diverse_sample_size`: Number of papers to show during onboarding
- `gp_num_samples`: Number of GP samples for uncertainty estimation
- `n_nearest_embeddings`: Number of nearest embeddings to use for prediction

## Usage

Paper Recommender offers two interfaces: a command-line interface (CLI) for terminal users and a graphical user interface (GUI) for those who prefer a more visual experience.

### Command-line Interface

#### First-time Use

On first run, the system will automatically start the onboarding process:

```bash
paper-recommender
```

#### Onboarding

The onboarding process helps the system learn your preferences by asking you to rate papers:

```bash
paper-recommender --onboard
```

During onboarding:
1. You'll be presented with papers selected using different strategies (random and diverse)
2. Rate papers on a scale of 1-5
3. The system will use these ratings to build a recommendation model

#### Getting Recommendations

After onboarding, you can get paper recommendations:

```bash
paper-recommender --recommend
```

The recommendation process:
1. Fetches recent papers from arXiv
2. Uses the trained model to predict your ratings
3. Presents papers with the highest predicted ratings
4. Allows you to rate the recommended papers to improve future recommendations

#### Bootstrapping the Model

If you want to retrain the recommendation model with your latest ratings:

```bash
paper-recommender --bootstrap
```

#### Command-line Options

```
usage: paper-recommender [-h] [--onboard] [--recommend] [--bootstrap]
                         [--config CONFIG] [--chroma-db-path CHROMA_DB_PATH]
                         [--model-path MODEL_PATH]
                         [--embedding-cache-path EMBEDDING_CACHE_PATH]
                         [--embedding-provider {ollama,openai}]
                         [--openai-api-key OPENAI_API_KEY]
                         [--openai-embedding-model OPENAI_EMBEDDING_MODEL]
                         [--exploration-weight EXPLORATION_WEIGHT]
                         [--max-samples MAX_SAMPLES]
                         [--period-hours PERIOD_HOURS]
                         [--random-sample-size RANDOM_SAMPLE_SIZE]
                         [--diverse-sample-size DIVERSE_SAMPLE_SIZE]
                         [--num-recommendations NUM_RECOMMENDATIONS]

Paper Recommender

optional arguments:
  -h, --help            show this help message and exit
  --onboard             Run onboarding process
  --recommend           Run recommendation process
  --bootstrap           Bootstrap the recommendation model
  --config CONFIG       Path to custom config file
  --chroma-db-path CHROMA_DB_PATH
                        Path to ChromaDB directory
  --model-path MODEL_PATH
                        Path to model pickle file
  --embedding-cache-path EMBEDDING_CACHE_PATH
                        Path to embedding cache file
  --embedding-provider {ollama,openai}
                        Embedding provider to use
  --openai-api-key OPENAI_API_KEY
                        OpenAI API key (only used with openai provider)
  --openai-embedding-model OPENAI_EMBEDDING_MODEL
                        OpenAI embedding model (only used with openai provider)
  --exploration-weight EXPLORATION_WEIGHT
                        Exploration weight for recommendations
  --max-samples MAX_SAMPLES
                        Maximum number of samples for similarity search
  --period-hours PERIOD_HOURS
                        Time period in hours for paper retrieval
  --random-sample-size RANDOM_SAMPLE_SIZE
                        Number of random papers to select during onboarding
  --diverse-sample-size DIVERSE_SAMPLE_SIZE
                        Number of diverse papers to select during onboarding
  --num-recommendations NUM_RECOMMENDATIONS
                        Number of recommendations to show
```

### Graphical User Interface

Paper Recommender also provides a web-based graphical interface for a more interactive experience.

#### Launching the GUI

To start the graphical interface:

```bash
paper-recommender-ui
```

By default, this will open a Chrome window with the Paper Recommender interface. If Chrome is not available, it will fall back to your default browser.

#### GUI Options

The GUI application accepts several command-line options to customize its behavior:

```
usage: paper-recommender-ui [-h] [--mode {chrome,electron,browser,default}] [--host HOST] [--port PORT] [--no-block]

Paper Recommender UI

optional arguments:
  -h, --help            show this help message and exit
  --mode {chrome,electron,browser,default}
                        Mode to start Eel in (default: chrome)
  --host HOST           Host to bind to (default: localhost)
  --port PORT           Port to bind to (default: 8000)
  --no-block            Don't block the main thread
```

#### GUI Features

The graphical interface provides all the functionality of the command-line version with a more user-friendly experience:

1. **Onboarding**: Rate papers through an intuitive interface to help the system learn your preferences
2. **Recommendations**: View and rate recommended papers with a clean, visual layout
3. **Search**: Search through your paper library using semantic search
4. **Visualization**: View visualizations of the recommendation model to understand how it works
5. **Configuration**: Easily adjust system settings through a configuration panel

The GUI also provides additional features not available in the CLI version:
- Adding custom papers manually
- Viewing your paper library with filtering options
- Interactive data visualizations of the Gaussian Process model

## How It Works

### Data Sources

Paper Recommender fetches recent papers from arXiv based on the configured time period. It uses the arXiv API to retrieve paper titles, abstracts, and links.

### Embeddings

The system supports two embedding providers:

1. **Ollama with nomic-embed-text (Default)**: Uses the nomic-embed-text model through Ollama to generate vector embeddings locally.

2. **OpenAI Embeddings**: Alternatively, you can use OpenAI's embedding models by setting `embedding_provider` to "openai" and providing your API key.

These embeddings capture the semantic meaning of papers, allowing the system to find similar papers.

### Vector Store

Embeddings and ratings are stored in a ChromaDB vector database, enabling efficient similarity search.

### Recommendation Algorithm

The recommendation system uses a sophisticated Gaussian Process Regression (GPR) model that:

1. Learns to predict the variance between ratings as a function of similarity
2. Uses this variance to weight similar papers when making predictions
3. Samples the GP model to estimate uncertainty
4. Uses N-sigma confidence intervals to balance exploration and exploitation

This approach provides more robust recommendations with better uncertainty estimates compared to directly predicting ratings.

## Troubleshooting

### Common Issues

- **Ollama Connection Error**: Ensure the Ollama server is running with `ollama serve`
- **Missing Model**: If you get an error about the nomic-embed-text model, run `ollama pull nomic-embed-text`
- **OpenAI API Key Issues**: If using the OpenAI embedding provider, ensure your API key is valid and has sufficient quota
- **No Recommendations**: You may need to onboard more papers before getting recommendations

### Reporting Issues

If you encounter any bugs or have feature requests, please open an issue on the GitHub repository.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
