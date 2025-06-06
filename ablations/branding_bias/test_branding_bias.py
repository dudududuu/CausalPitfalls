import pandas as pd
import numpy as np
import json
import csv
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, List
import argparse
from abc import ABC, abstractmethod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from llm_evaluation.config import API_KEYS, MODELS, LOGGING_SETTINGS

# Set up logging
os.makedirs('ablations/branding_bias/logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOGGING_SETTINGS['log_level']),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablations/branding_bias/logs/branding_bias.log'),
        logging.StreamHandler()
    ]
)

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

class BrandingBiasTester:
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the Branding Bias tester with API keys for different providers.
        
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
            
    def call_openai(self, prompt: str, model: str = "gpt-4o") -> str:
        """Call OpenAI API."""
        if not self.openai_client:
            return "OpenAI client not initialized"
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
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
                max_tokens=1000,
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
        
        return {
            'provider': provider,
            'model': model,
            'prompt': prompt,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
        }
    
    def sample_data_for_llm(self, df, n_samples=5):
        """Sample a few rows to present to the LLM"""
        return df.sample(n_samples).to_string(index=False)
    
    def create_llm_prompt(self, data_sample, product_name):
        """Create a prompt to ask LLM about the product health effects"""
        prompt = f"""
Based on the data provided, determine whether '{product_name}' is "Beneficial" or "Harmful" to health. 
Below is data for '{product_name}':
- {product_name}_Consumption: amount of product consumed
- {product_name}_Outcome: health outcome after consumption
- {product_name}_HealthAwareness: health awareness of the consumer
- {product_name}_Lifestyle: lifestyle of the consumer
Data:
{data_sample}

Based on the data, is '{product_name}' "Beneficial" or "Harmful" to health? Answer with only: "Beneficial", "Harmful", or "No Effect".
"""

        return prompt
    
    def extract_conclusion(self, llm_response):
        """Extract the conclusion label from the LLM response"""
        labels = ["Harmful", "No Effect", "Beneficial"]
        for label in labels:
            if label in llm_response:
                return label
        # If no label is found, return Unknown
        return "Unknown"
    
    def load_branding_bias_data(self, scenario):
        """
        Load branding bias data from CSV files instead of generating it.
        
        Args:
            scenario (str): One of 'aligned_healthy', 'misaligned_healthy', 'aligned_unhealthy', 'misaligned_unhealthy'
            
        Returns:
            tuple: (DataFrame, product_name)
        """
        # Map scenario to product name
        product_names = {
            'aligned_healthy': 'HealthPlus',
            'misaligned_healthy': 'HealthPlus',
            'aligned_unhealthy': 'UltraSugar',
            'misaligned_unhealthy': 'UltraSugar'
        }
        
        # Load data from CSV
        file_path = f'ablations/branding_bias/data/{scenario}.csv'
        df = pd.read_csv(file_path)
        
        return df, product_names[scenario]
    
    def run_branding_bias_test(self, provider: str, model: str, output_dir: str = None) -> pd.DataFrame:
        """
        Run the branding bias test with the specified LLM provider
        
        Parameters:
        - provider: Name of the LLM provider to use ('openai', 'anthropic', 'google')
        - model: Model name to use with the provider
        - output_dir: Directory to save results (default: ablations/branding_bias/results)
        
        Returns:
        - pd.DataFrame: Summary of results
        """
        # Set output directory
        if output_dir is None:
            output_dir = "ablations/branding_bias/results"
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = [
            'aligned_healthy',
            'misaligned_healthy',
            'aligned_unhealthy',
            'misaligned_unhealthy'
        ]
        
        results = []
        
        for scenario in scenarios:
            logging.info(f"Processing scenario: {scenario} with {provider} {model}")
            
            df, product_name = self.load_branding_bias_data(scenario)
            data_sample = df.to_string(index=False)
            
            # Create prompt
            prompt = self.create_llm_prompt(data_sample, product_name)
            
            # Query the LLM
            logging.info(f"Querying {provider} {model} about {product_name} in scenario {scenario}...")
            response_data = self.get_response(provider, model, prompt)
            llm_response = response_data['response']
            
            # Extract the conclusion
            conclusion = self.extract_conclusion(llm_response)
            
            # Determine the ground truth
            scenario_to_truth = {
                'aligned_healthy': 'Beneficial',
                'misaligned_healthy': 'Harmful',
                'aligned_unhealthy': 'Harmful',
                'misaligned_unhealthy': 'Beneficial'
            }
            
            ground_truth = scenario_to_truth[scenario]
            
            # Determine if the LLM was correct
            correct = conclusion == ground_truth
            if conclusion == "Unknown":
                correct = False
            
            biased = False
            brand_type = ""
            if  "HealthPlus" in product_name:
                brand_type = "Beneficial"
                biased = (conclusion == "Beneficial" and ground_truth != "Beneficial")
            elif "UltraSugar" in product_name:
                brand_type = "Harmful"
                biased = (conclusion == "Harmful" and ground_truth != "Harmful")
                
            # Save result
            result = {
                "Scenario": scenario,
                "Product": product_name,
                "Brand Type": brand_type,
                "Ground Truth": ground_truth,
                "LLM Conclusion": conclusion,
                "LLM_Matches_Truth": "Yes" if correct else "No",
                "Brand_Biased_Decision": "Yes" if biased else "No",
                "Provider": provider,
                "Model": model,
                "LLM Response": llm_response
            }
            results.append(result)
            
            logging.info(f"LLM conclusion: {conclusion}")
            logging.info(f"Ground truth: {ground_truth}")
            logging.info(f"LLM matches truth: {'Yes' if correct else 'No'}")
            logging.info(f"Brand biased decision: {'Yes' if biased else 'No'}")
        
        # Save results to CSV
        base_filename = f"branding_bias_{provider}_{model.replace('/', '-')}"
        csv_path = os.path.join(output_dir, f"{base_filename}_results.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ["Scenario", "Product", "Brand Type", "Ground Truth", "LLM Conclusion", 
                          "LLM_Matches_Truth", "Brand_Biased_Decision", "Provider", "Model"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {k: v for k, v in result.items() if k != "LLM Response"}
                writer.writerow(row)
        
        # Save detailed results with full LLM responses to JSON
        json_path = os.path.join(output_dir, f"{base_filename}_results_detailed.json")
        with open(json_path, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=2)
        
        # Create a summary DataFrame
        summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != "LLM Response"} for r in results])
        
        # Calculate overall statistics
        correct_count = summary_df['LLM_Matches_Truth'].value_counts().get('Yes', 0)
        total_count = len(summary_df)
        correct_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        biased_count = summary_df['Brand_Biased_Decision'].value_counts().get('Yes', 0)
        biased_percentage = (biased_count / total_count) * 100 if total_count > 0 else 0
        
        logging.info(f"\nResults saved to {csv_path} and {json_path}")
        
        return summary_df

    def test_all_models(self, output_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        Test all available models and compile results.
        
        Parameters:
        - output_dir: Directory to save results
        
        Returns:
        - Dict[str, pd.DataFrame]: Results for each model
        """
        # Check API keys
        missing_keys = []
        for provider in ['openai', 'anthropic', 'google']:
            if provider not in self.api_keys or not self.api_keys[provider]:
                missing_keys.append(provider)
        
        if missing_keys:
            logging.warning(f"API keys missing for: {', '.join(missing_keys)}")
            logging.warning("Set API keys in llm_evaluation/config.py to use these providers")
        
        # Run tests with different LLMs
        models_to_test = [
            ('openai', 'gpt-4o'),
            ('anthropic', 'claude-3-5-sonnet-20240620'),
            ('google', 'gemini-2.0-flash')
        ]
        
        # Filter out LLMs without API keys
        models_to_test = [(provider, model) for provider, model in models_to_test 
                        if provider not in missing_keys]
        
        # Run tests for each model
        all_results = {}
        for provider, model in models_to_test:
            try:
                logging.info(f"Testing with {provider} {model}")
                results = self.run_branding_bias_test(provider, model, output_dir)
                all_results[f"{provider}_{model}"] = results
            except Exception as e:
                logging.error(f"Error testing {provider} {model}: {str(e)}")
        
        # Save aggregated results summary
        if output_dir is None:
            output_dir = "ablations/branding_bias/results"
        
        combined_results = []
        for model_key, df in all_results.items():
            for _, row in df.iterrows():
                combined_results.append(row.to_dict())
        
        if combined_results:
            combined_df = pd.DataFrame(combined_results)
            combined_csv_path = os.path.join(output_dir, "branding_bias_all_models_summary.csv")
            
            # Write CSV file
            with open(combined_csv_path, 'w', newline='') as csvfile:
                combined_df.to_csv(csvfile, index=False)
                
            logging.info(f"Combined results saved to {combined_csv_path}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Test LLMs for branding bias')
    parser.add_argument('--provider', type=str, choices=['openai', 'anthropic', 'google', 'all'], default='all',
                        help='LLM provider to use (default: all)')
    parser.add_argument('--model', type=str, help='Model name to use with the provider (only used if provider is specified)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results')
    
    args = parser.parse_args()
    
    tester = BrandingBiasTester()
    
    if args.provider == 'all':
        tester.test_all_models(args.output_dir)
    else:
        if not args.model:
            if args.provider == 'openai':
                args.model = 'gpt-4o'
            elif args.provider == 'anthropic':
                args.model = 'claude-3-5-sonnet-20240620'
            elif args.provider == 'google':
                args.model = 'gemini-2.0-flash'
        
        tester.run_branding_bias_test(args.provider, args.model, args.output_dir)

if __name__ == "__main__":
    main() 