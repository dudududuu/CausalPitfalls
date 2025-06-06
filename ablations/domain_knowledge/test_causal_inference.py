import os
import json
import pandas as pd
import sys
from typing import Dict, Any, List
import time
import logging
from datetime import datetime
from collections import defaultdict
import html
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from llm_evaluation.config import API_KEYS, MODELS, LOGGING_SETTINGS

# Set up logging
os.makedirs('ablations/domain_knowledge/logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOGGING_SETTINGS['log_level']),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablations/domain_knowledge/logs/causal_inference.log'),
        logging.StreamHandler()
    ]
)

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai


class CausalInferenceTester:
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the Causal Inference tester with API keys for different providers.
        
        Args:
            api_keys (Dict[str, str], optional): Dictionary containing API keys for different providers.
                If None, uses keys from config file.
        """
        self.api_keys = api_keys or API_KEYS
        self._setup_apis()
        
    def _setup_apis(self):
        """Set up API clients for different providers."""
        if 'openai' in self.api_keys and OpenAI:
            self.openai_client = OpenAI(api_key=self.api_keys['openai'])
        else:
            self.openai_client = None
            
        if 'anthropic' in self.api_keys and Anthropic:
            self.anthropic_client = Anthropic(api_key=self.api_keys['anthropic'])
        else:
            self.anthropic_client = None
            
        if 'google' in self.api_keys and genai:
            genai.configure(api_key=self.api_keys['google'])
        
    def load_data(self, data_type: str, abstract_names: bool = False, pure_noise: bool = False) -> pd.DataFrame:
        """
        Load dataset for evaluation.
        
        Args:
            data_type (str): Type of data to load ('confounder' or 'mediator')
            abstract_names (bool): Whether to load version with abstract column names
            pure_noise (bool): Whether to load the pure_noise (random) data version
            
        Returns:
            pd.DataFrame: Loaded data
        """
        filename_parts = [f"{data_type}_data"]
        if pure_noise:
            filename_parts.append("pure_noise")
        if abstract_names:
            filename_parts.append("abstract")
        
        filename = "_".join(filename_parts) + ".csv"
        data_path = os.path.join('ablations/domain_knowledge/data', filename)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}. Run data_generator.py first.")
        
        return pd.read_csv(data_path)
    
    def call_openai(self, prompt: str, model: str = "gpt-4o") -> str:
        """Call OpenAI API."""
        if not self.openai_client:
            return "OpenAI client not initialized"
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"
    
    def call_anthropic(self, prompt: str, model: str = "claude-3-5-sonnet-20240620") -> str:
        """Call Anthropic API."""
        if not self.anthropic_client:
            return "Anthropic client not initialized"
        
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API error: {str(e)}")
            return f"Error: {str(e)}"
    
    def call_google(self, prompt: str, model: str = "gemini-2.0-flash") -> str:
        """Call Google AI API."""
        if not genai:
            return "Google Generative AI not initialized"
        
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Google AI API error: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_causal_prompt(self, data_type: str, abstract: bool, data: pd.DataFrame) -> str:
        """
        Generate a prompt about causal relationships for a given dataset.
        
        Args:
            data_type (str): Type of data ('confounder' or 'mediator')
            abstract (bool): Whether to use abstract variable names
            data (pd.DataFrame): The dataset to include in the prompt
            
        Returns:
            str: The prompt
        """
        # Get column names
        col_names = list(data.columns)
        
        # Convert data to string format
        data_sample = data.to_string(index=False)
        
        # Define prompts based on data type and abstraction
        if data_type == 'confounder':
            if not abstract:
                question = "Determine if there is a causal relationship between ice cream sales and drowning incidents. Construct a Directed Acyclic Graph (DAG) representing the relevant causal relationships. Justify your conclusion based on numerical analysis from the provided dataset, but do not include code or calculation details in your response."
                prompt = f"Given the dataset with columns {col_names}:\n{data_sample}"

        full_prompt = prompt + question
        return full_prompt
    
    def get_response(self, provider: str, model: str, prompt: str) -> Dict[str, Any]:
        """
        Get a response from the specified model for a given prompt.
        
        Args:
            provider (str): Provider name ('openai', 'anthropic', 'google')
            model (str): Model name
            prompt (str): The prompt to send
            
        Returns:
            Dict[str, Any]: Response data
        """
        start_time = time.time()
        
        if provider == 'openai':
            response_text = self.call_openai(prompt, model)
        elif provider == 'anthropic':
            response_text = self.call_anthropic(prompt, model)
        elif provider == 'google':
            response_text = self.call_google(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        duration = time.time() - start_time
        
        # Create a simplified prompt without data sample for storage
        simplified_prompt = self._simplify_prompt_for_storage(prompt)
        
        return {
            'provider': provider,
            'model': model,
            'prompt': simplified_prompt,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
        }
    
    def _simplify_prompt_for_storage(self, prompt: str) -> str:
       
        if "Given the dataset with columns" in prompt:
            data_start_idx = prompt.find("Given the dataset with columns")
            question_indicators = ["Determine if there is", "Analyze this dataset"]
            question_start_idx = float('inf')
            for indicator in question_indicators:
                idx = prompt.find(indicator)
                if idx != -1 and idx < question_start_idx:
                    question_start_idx = idx
            
            if question_start_idx < float('inf'):
                # Get the column names part
                columns_part = prompt[data_start_idx:prompt.find("\n\n", data_start_idx)]
                question_part = prompt[question_start_idx:]
                return f"{columns_part}\n\n[DATA SAMPLE REMOVED]\n\n{question_part}"
        return prompt
    
    def test_all_datasets(self, provider: str, model: str) -> Dict[str, Dict[str, Any]]:
        """
        Test all dataset variations and get responses.
        
        Args:
            provider (str): Provider name ('openai', 'anthropic', 'google')
            model (str): LLM name
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping dataset names to response data
        """
        results = {}
        
        # Initialize results directory
        os.makedirs('ablations/domain_knowledge/results/causal_inference', exist_ok=True)
        
        # Define test configurations
        test_configs = [
            (False, False),  # real
            (False, True),   # pure_noise
        ]
        
        data_type = 'confounder'
        
        for abstract, pure_noise in test_configs:
            try:
                # Load the dataset
                data = self.load_data(data_type, abstract_names=abstract, pure_noise=pure_noise)
                
                # Generate dataset key for results
                dataset_key = f"{'pure_noise' if pure_noise else 'real'}"
                
                # Generate prompt
                prompt = self.generate_causal_prompt(data_type, abstract, data)
                
                # Get model response
                logging.info(f"Testing {dataset_key} with {provider} {model}")
                response_data = self.get_response(provider, model, prompt)
                results[dataset_key] = response_data
                
                # Save individual result 
                output_path = f"ablations/domain_knowledge/results/causal_inference/{dataset_key}_{provider}_{model.replace('/', '-')}.json"
                with open(output_path, 'w') as f:
                    json.dump(response_data, f, indent=2)
                    
                logging.info(f"Saved result to {output_path}")
            except Exception as e:
                logging.error(f"Error testing {data_type} (abstract={abstract}, pure_noise={pure_noise}): {str(e)}")
                dataset_key = f"{'pure_noise' if pure_noise else 'real'}"
                results[dataset_key] = {"error": str(e)}
        
        return results

    
def main():
    # Check API keys 
    missing_keys = []
    for provider in ['openai', 'anthropic', 'google']:
        if provider not in API_KEYS or not API_KEYS[provider]:
            missing_keys.append(provider)
    
    if missing_keys:
        logging.warning(f"API keys missing for: {', '.join(missing_keys)}")
    
    # Initialize tester
    tester = CausalInferenceTester()
    
    # Run tests with different LLMs
    models_to_test = [
        ('openai', 'gpt-4o'),
        ('anthropic', 'claude-3-5-sonnet-20240620'),
        ('google', 'gemini-2.0-flash')
    ]
    
    # Filter out models without API keys
    models_to_test = [(provider, model) for provider, model in models_to_test 
                     if provider not in missing_keys]
    
    # Run tests for each model
    all_results = {}
    for provider, model in models_to_test:
        try:
            logging.info(f"Testing with {provider} {model}")
            results = tester.test_all_datasets(provider, model)
            all_results[f"{provider}_{model}"] = results
        except Exception as e:
            logging.error(f"Error testing {provider} {model}: {str(e)}")
    
    # Save aggregated results
    output_path = f"ablations/domain_knowledge/results/causal_inference_all_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"All results saved to {output_path}")


if __name__ == "__main__":
    main() 