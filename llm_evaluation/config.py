import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
API_KEYS = {
    'openai': os.getenv('OPENAI_API_KEY'),
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'google': os.getenv('GOOGLE_API_KEY')
}

# Model configurations
MODELS = {
    'openai': 'gpt-4o',
    'anthropic': 'claude-3-opus-20240229',
    'google': 'gemini-pro'
}

# Evaluation settings
EVALUATION_SETTINGS = {
    'data_sample_size': 100,
    'results_dir': 'llm_evaluation/results'
}

# Logging settings
LOGGING_SETTINGS = {
    'log_file': 'llm_evaluation.log',
    'log_level': 'INFO'
} 