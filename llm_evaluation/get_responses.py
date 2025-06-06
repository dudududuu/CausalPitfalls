import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from typing import List, Dict, Any
import openai
from anthropic import Anthropic
import google.generativeai as genai
from datetime import datetime
import logging
import subprocess
import re
from config import API_KEYS, LOGGING_SETTINGS
import codecs
from pathlib import Path
from openai import OpenAI
import time
import argparse
from llm_evaluation.code_execution_utils import execute_code_with_dependencies

# Configure logging with timestamps and proper formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# API configuration
base_url = "https://api.deepseek.com"

def save_direct_analysis(question_dir: Path, direct_prompt: str, direct_response: str):
    """Save direct analysis with proper encoding."""
    with codecs.open(os.path.join(question_dir, 'direct_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Prompt:\n{direct_prompt}\n\nResponse:\n{direct_response}")

def save_generated_code(question_dir: Path, extracted_code: str):
    """Save generated code with proper encoding."""
    with codecs.open(os.path.join(question_dir, 'generated_code.py'), 'w', encoding='utf-8') as f:
        f.write(extracted_code)

def save_execution_results(question_dir: Path, execution_results: str):
    """Save execution results with proper encoding."""
    with codecs.open(os.path.join(question_dir, 'execution_results.txt'), 'w', encoding='utf-8') as f:
        f.write(execution_results)

def save_results_analysis(question_dir: Path, results_prompt: str, results_response: str):
    """Save results analysis with proper encoding."""
    with codecs.open(os.path.join(question_dir, 'results_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Prompt:\n{results_prompt}\n\nResponse:\n{results_response}")

def save_error_info(question_dir: Path, error_msg: str, code: str = None):
    """Save error information with proper encoding."""
    with codecs.open(os.path.join(question_dir, 'error.txt'), 'w', encoding='utf-8') as f:
        if code:
            f.write(f"Error: {error_msg}\n\nCode:\n{code}")
        else:
            f.write(f"Error: {error_msg}")

def save_evaluation(scores: List[int], rubric: Dict[str, Any], challenge_name: str, dataset_name: str, 
                   difficulty: str, llm_name: str, protocol_type: str, response_text: str, file_path: Path):
    """Save the evaluation results in the results directory."""
    output_dir = Path(f"challenges/{challenge_name}/results/{llm_name}/{dataset_name}/run/{difficulty}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluation = {
        "metadata": {
            "challenge": challenge_name,
            "dataset": dataset_name,
            "llm": llm_name,
            "protocol_type": protocol_type,
            "max_score": rubric["max_score"],
            "file_path": str(file_path)
        },
        "scores": dict(zip([c["id"] for c in rubric["criteria"]], scores)),
        "total_score": sum(scores),
        "response": response_text
    }
    
    output_path = output_dir / f"{llm_name}_{protocol_type}_evaluation.json"
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    return evaluation

class LLMEvaluator:
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the LLM evaluator with API keys for different providers.
        
        Args:
            api_keys (Dict[str, str], optional): Dictionary containing API keys for different providers.
                If None, uses keys from config file.
        """
        self.api_keys = api_keys or API_KEYS
        self._setup_apis()
        
    def _setup_apis(self):
        """Set up API clients for different providers."""
        if 'openai' in self.api_keys:
            openai.api_key = self.api_keys['openai']
            self.openai_client = OpenAI()
        if 'anthropic' in self.api_keys:
            self.anthropic_client = Anthropic(api_key=self.api_keys['anthropic'])
            
        if 'google' in self.api_keys:
            genai.configure(api_key=self.api_keys['google'])
            
        if 'deepseek' in self.api_keys:
            self.deepseek_client = OpenAI(
                api_key=self.api_keys['deepseek'],
                base_url="https://api.deepseek.com"
            )
    
    def load_challenge_data(self, challenge_name: str, dataset_name: str) -> tuple:
        """
        Load questions, answer key, and data for a specific challenge.
        
        Args:
            challenge_name (str): Name of the challenge (e.g., 'simpson_paradox')
            dataset_name (str): Name of the dataset
            
        Returns:
            tuple: (questions, answer_key, data) where questions is a list of questions,
                  answer_key is a dictionary of answer key items, and data is a pandas DataFrame
        """
        challenge_path = os.path.join('challenges', challenge_name)
        
        # Load questions
        questions_path = os.path.join(challenge_path, 'questions.json')
        with open(questions_path, 'r') as f:
            questions = json.load(f)
            
        # Load answer key
        answer_key_path = os.path.join(challenge_path, 'answer_key.json')
        with open(answer_key_path, 'r') as f:
            answer_key = json.load(f)
            
        # Load data
        data_path = os.path.join(challenge_path, f'{dataset_name}.csv')
        data = pd.read_csv(data_path)
        
        return questions, answer_key, data
    
    def call_openai(self, prompt: str, model: str = "gpt-4o") -> str:
        """Call OpenAI API."""
        try:
            if 'o4-mini' in model:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="medium"
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=2048,
                )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise   
    
    def call_anthropic(self, prompt: str, model: str = "claude-3-opus-20240229") -> str:
        """Call Anthropic API."""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API error: {str(e)}")
            raise 
    
    def call_google(self, prompt: str, model: str = "gemini-pro") -> str:
        """Call Google AI API."""
        try:
            model = genai.GenerativeModel(model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Google AI API error: {str(e)}")
            raise 

    def call_deepseek(self, prompt: str, model: str = "deepseek-chat") -> str:
        """Call Deepseek API."""
        try:
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
                stream=False,

            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Deepseek API error: {str(e)}")
            raise
    
    def get_response(self, provider: str, model: str, prompt: str) -> Dict[str, Any]:
        """
        Get a response from the specified model for a given prompt.
        
        Args:
            provider (str): Provider name ('openai', 'anthropic', 'google', 'deepseek')
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
        elif provider == 'deepseek':
            response_text = self.call_deepseek(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        duration = time.time() - start_time
        
        # Create a simplified prompt without data sample for storage only
        simplified_prompt = self._simplify_prompt_for_storage(prompt)
        
        return {
            'provider': provider,
            'model': model,
            'prompt': simplified_prompt,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
        }

    def evaluate_model(self, 
                      challenge_name: str, 
                      dataset_name: str,
                      provider: str,
                      model: str,
                      difficulty: str) -> Dict[str, Any]:
        """
        Evaluate any LLM on a specific challenge with two types of prompts:
        1. Direct Prompt: Direct analysis with data
        2. Code-Assisted Prompt: Code generation and execution
        
        Args:
            challenge_name (str): Name of the challenge
            dataset_name (str): Name of the dataset
            provider (str): Provider name ('openai', 'anthropic', or 'google')
            model (str): Model name to use
            difficulty (str): Current difficulty level being processed
            
        Returns:
            Dict[str, Any]: Results of the evaluation
        """
        # Load challenge data
        questions, answer_key, data = self.load_challenge_data(challenge_name, dataset_name)
        
        # Find the question for the current difficulty
        current_question = next((q for q in questions if q['difficulty'] == difficulty), None)
        if not current_question:
            raise ValueError(f"No question found for difficulty: {difficulty}")
            
        # Prepare data sample (shuffle and first 100 rows)
        data = data.sample(frac=1).reset_index(drop=True)
        data_sample = data.head(100).to_string()
        
        # Create results directory structure
        challenge_path = os.path.join('challenges', challenge_name)
        results_dir = os.path.join(challenge_path, 'results', model, dataset_name, 'run')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create summary file
        summary = {
            'challenge_name': challenge_name,
            'dataset_name': dataset_name,
            'provider': provider,
            'model': model,
            'questions': []
        }
        
        # Choose the appropriate API call method
        if provider == 'openai':
            api_call = lambda prompt: self.call_openai(prompt, model)
        elif provider == 'anthropic':
            api_call = lambda prompt: self.call_anthropic(prompt, model)
        elif provider == 'google':
            api_call = lambda prompt: self.call_google(prompt, model)
        elif provider == 'deepseek':
            api_call = lambda prompt: self.call_deepseek(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Create question directory with difficulty level
        question_dir = os.path.join(results_dir, difficulty)
        os.makedirs(question_dir, exist_ok=True)
        
        # Create metadata file for this question
        metadata = {
            'challenge_name': challenge_name,
            'dataset_name': dataset_name,
            'provider': provider,
            'model': model,
            'difficulty': difficulty,
            'question': current_question['prompt']
        }
        
        with codecs.open(os.path.join(question_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Type 1: Direct Prompt
        direct_prompt = f"""Question: {current_question['prompt']}

Data:
{data_sample}

Provide a direct analysis of the question and data without any code.
"""
        
        logging.info(f"Processing {provider} {model} direct prompt for {difficulty} question: {current_question['prompt'][:50]}...")
        direct_response = api_call(direct_prompt)
        
        # Save direct prompt
        save_direct_analysis(question_dir, direct_prompt, direct_response)
        
        # Type 2: Code generation prompt
        data_path = os.path.join(challenge_path, f'{dataset_name}.csv')
        code_prompt = f"""Question: {current_question['prompt']}

The data is available at: {data_path}, data column names are: {data.columns.tolist()}, sample data (first 10 rows): {data.head(10).to_string()}

Generate code to provide necessary numbers. Provide Python code only:"""
        
        logging.info(f"Processing {provider} {model} code generation for {difficulty} question: {current_question['prompt'][:50]}...")
        code_response = api_call(code_prompt)
        # Extract code from markdown blocks
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', code_response, re.DOTALL)
        if not code_blocks:
            error_msg = "No Python code block found in the response"
            logging.error(error_msg)
            # Save error information
            save_error_info(question_dir, error_msg)
            extracted_code = ""  # Empty code
            execution_results = "NULL"
            
            # Add to summary without raising error
            summary['questions'].append({
                'difficulty': difficulty,
                'question': current_question['prompt'],
                'files': {
                    'metadata': 'metadata.json',
                    'direct_analysis': 'direct_analysis.txt',
                    'generated_code': None,
                    'execution_results': None,
                    'results_analysis': None,
                    'error': 'error.txt',
                    'warning': None
                }
            })
            
            return summary
        extracted_code = code_blocks[0].strip()
        
        if not extracted_code:
            error_msg = "Empty code block found"
            logging.error(error_msg)
            # Save error information
            save_error_info(question_dir, error_msg)
            execution_results = "NULL"
            
            # Add to summary without raising error
            summary['questions'].append({
                'difficulty': difficulty,
                'question': current_question['prompt'],
                'files': {
                    'metadata': 'metadata.json',
                    'direct_analysis': 'direct_analysis.txt',
                    'generated_code': None,
                    'execution_results': None,
                    'results_analysis': None,
                    'error': 'error.txt',
                    'warning': None
                }
            })
            
            return summary
        
        # Save generated code
        save_generated_code(question_dir, extracted_code)
        
        # Execute the generated code and get results
        try:
            execution_results = execute_code_with_dependencies(
                extracted_code=extracted_code,
                question_dir=question_dir
            )
            
        except Exception as e:
            error_msg = f"Error during code execution: {str(e)}"
            logging.error(error_msg)
            # Save error information
            save_error_info(question_dir, error_msg)
            execution_results = "NULL"
        
        # Send results back to LLM for analysis
        results_prompt = f"""Question: {current_question['prompt']}

Code:
{extracted_code}

Results:
{execution_results}
"""
        
        logging.info(f"Processing {provider} {model} Code-Assisted Prompt for {difficulty} question: {current_question['prompt'][:50]}...")
        try:
            results_response = api_call(results_prompt)
            
            # Save code-assisted analysis
            save_results_analysis(question_dir, results_prompt, results_response)
        except Exception as e:
            error_msg = f"Error during results analysis: {str(e)}"
            logging.error(error_msg)
            # Save error information
            save_error_info(question_dir, error_msg)
            results_response = None
        
        # Add to summary
        summary['questions'].append({
            'difficulty': difficulty,
            'question': current_question['prompt'],
            'files': {
                'metadata': 'metadata.json',
                'direct_analysis': 'direct_analysis.txt',
                'generated_code': 'generated_code.py',
                'execution_results': 'execution_results.txt',
                'results_analysis': 'results_analysis.txt' if results_response else None,
                'error': 'error.txt' if os.path.exists(os.path.join(question_dir, 'error.txt')) else None,
                'warning': 'warning.txt' if os.path.exists(os.path.join(question_dir, 'warning.txt')) else None
            }
        })
        
        # Save summary
        with codecs.open(os.path.join(results_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Create error summary
        error_summary = {
            'challenge_name': challenge_name,
            'dataset_name': dataset_name,
            'provider': provider,
            'model': model,
            'errors': []
        }
        
        # Collect all errors
        for question in summary['questions']:
            if question['files']['error']:
                error_path = os.path.join(results_dir, question['difficulty'], question['files']['error'])
                try:
                    with open(error_path, 'r') as f:
                        error_content = f.read()
                    
                    error_summary['errors'].append({
                        'difficulty': question['difficulty'],
                        'question': question['question'],
                        'error_message': error_content
                    })
                except Exception as e:
                    logging.error(f"Failed to read error file {error_path}: {str(e)}")
        
        # Save error summary
        with codecs.open(os.path.join(results_dir, 'error_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(error_summary, f, indent=2)
            
        logging.info(f"Results saved to {results_dir}")
        return summary
    

def check_model_results_exist(challenge_name: str, dataset_name: str, model: str, provider: str, difficulty: str) -> bool:
    """Check if results already exist for a given challenge, dataset, model, and provider."""
    # Use forward slashes for path construction
    results_dir = os.path.join('challenges', challenge_name, 'results', model, dataset_name, 'run').replace('\\', '/')
    logging.info(f"Checking results in: {results_dir} for difficulty: {difficulty}")
    if not os.path.exists(results_dir):
        logging.info(f"Results directory does not exist: {results_dir}")
        return False
        
    # Check only the current difficulty folder
    difficulty_dir = os.path.join(results_dir, difficulty).replace('\\', '/')
    logging.info(f"Checking difficulty directory: {difficulty_dir}")
    if not os.path.exists(difficulty_dir):
        logging.info(f"Difficulty directory does not exist: {difficulty_dir}")
        return False
        
    # Check for essential result files
    required_files = ['results_analysis.txt', 'execution_results.txt', 'generated_code.py']
    for file in required_files:
        file_path = os.path.join(difficulty_dir, file).replace('\\', '/')
        if not os.path.exists(file_path):
            logging.info(f"Required file does not exist: {file_path}")
            return False
        else:
            logging.info(f"Found required file: {file_path}")
    
    logging.info(f"All required files exist in {difficulty_dir}")
    return True

def main(challenge_name: str = None, dataset: str = None):
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY'),
        'deepseek': os.getenv('DEEPSEEK_API_KEY')
    }
    
    evaluator = LLMEvaluator(api_keys)
    
    # Get challenge directories
    challenges_dir = 'challenges'
    if challenge_name:
        # Only process the specified challenge if it exists
        if not os.path.isdir(os.path.join(challenges_dir, challenge_name)):
            logging.error(f"Challenge directory '{challenge_name}' not found")
            return
        challenge_dirs = [challenge_name]
    else:
        # Process all challenges
        challenge_dirs = [d for d in os.listdir(challenges_dir) 
                         if os.path.isdir(os.path.join(challenges_dir, d))]
    
    # Loop through each challenge
    for challenge_name in challenge_dirs:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing Challenge: {challenge_name}")
        logging.info(f"{'='*50}\n")
        
        challenge_path = os.path.join(challenges_dir, challenge_name)
        
        # Find all CSV files in the challenge directory (datasets)
        if dataset:
            # Only process the specified dataset
            datasets = [dataset]
        else:
            datasets = [f[:-4] for f in os.listdir(challenge_path) 
                       if f.endswith('.csv')]
        
        # Loop through each dataset
        for dataset_name in datasets:
            logging.info(f"\n{'-'*40}")
            logging.info(f"Processing Dataset: {dataset_name}")
            logging.info(f"{'-'*40}\n")
            
            try:
                # Load questions to get difficulties
                questions, answer_key, data = evaluator.load_challenge_data(challenge_name, dataset_name)
                
                # Process each difficulty level
                for question in questions:
                    difficulty = question['difficulty']
                    logging.info(f"\n{'='*20}")
                    logging.info(f"Processing question with difficulty: {difficulty}")
                    logging.info(f"Question: {question['prompt'][:100]}...")
                    logging.info(f"{'='*20}\n")
                    
                    # Test with OpenAI model
                    if not check_model_results_exist(challenge_name, dataset_name, 'o4-mini-2025-04-16', 'openai', difficulty):
                        logging.info(f"Processing OpenAI o4-mini-2025-04-16 for {challenge_name}/{dataset_name}/{difficulty}")
                        openai_results = evaluator.evaluate_model(
                            challenge_name=challenge_name,
                            dataset_name=dataset_name,
                            provider='openai',
                            model='o4-mini-2025-04-16',
                            difficulty=difficulty
                        )
                    else:
                        logging.info(f"Skipping OpenAI o4-mini-2025-04-16 for {challenge_name}/{dataset_name}/{difficulty} - results exist")

                    if not check_model_results_exist(challenge_name, dataset_name, 'gpt-4o', 'openai', difficulty):
                        logging.info(f"Processing OpenAI gpt-4o for {challenge_name}/{dataset_name}/{difficulty}")
                        openai_results = evaluator.evaluate_model(
                            challenge_name=challenge_name,
                            dataset_name=dataset_name,
                            provider='openai',
                            model='gpt-4o',
                            difficulty=difficulty
                        )
                    else:
                        logging.info(f"Skipping OpenAI gpt-4o for {challenge_name}/{dataset_name}/{difficulty} - results exist")

                    # Test with Deepseek model
                    if not check_model_results_exist(challenge_name, dataset_name, 'deepseek-chat', 'deepseek', difficulty):
                        logging.info(f"Processing Deepseek for {challenge_name}/{dataset_name}/{difficulty}")
                        deepseek_results = evaluator.evaluate_model(
                            challenge_name=challenge_name,
                            dataset_name=dataset_name,
                            provider='deepseek',
                            model='deepseek-chat',
                            difficulty=difficulty
                        )
                    else:
                        logging.info(f"Skipping Deepseek for {challenge_name}/{dataset_name}/{difficulty} - results exist")
                    
                    # Test with Anthropic model
                    if not check_model_results_exist(challenge_name, dataset_name, 'claude-3-5-sonnet-20240620', 'anthropic', difficulty):
                        logging.info(f"Processing Anthropic for {challenge_name}/{dataset_name}/{difficulty}")
                        anthropic_results = evaluator.evaluate_model(
                            challenge_name=challenge_name,
                            dataset_name=dataset_name,
                            provider='anthropic',
                            model='claude-3-5-sonnet-20240620',
                            difficulty=difficulty
                        )
                    else:
                        logging.info(f"Skipping Anthropic for {challenge_name}/{dataset_name}/{difficulty} - results exist")
                    
                    # Test with Google model
                    if not check_model_results_exist(challenge_name, dataset_name, 'gemini-2.0-flash', 'google', difficulty):
                        logging.info(f"Processing Google for {challenge_name}/{dataset_name}/{difficulty}")
                        google_results = evaluator.evaluate_model(
                            challenge_name=challenge_name,
                            dataset_name=dataset_name,
                            provider='google',
                            model='gemini-2.0-flash',
                            difficulty=difficulty
                        )
                    else:
                        logging.info(f"Skipping Google for {challenge_name}/{dataset_name}/{difficulty} - results exist")
                    
                    logging.info(f"Completed processing {challenge_name}/{dataset_name}/{difficulty}\n")
                
            except Exception as e:
                logging.error(f"Error processing {challenge_name}/{dataset_name}: {str(e)}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process challenge datasets with LLM evaluation")

    parser.add_argument("--challenge_name", type=str, default=None, help="Name of the challenge to process. If not provided, all challenges will be processed.")
    parser.add_argument("--dataset", help="Name of the specific dataset to process. If not provided, all datasets will be processed.")
    
    args = parser.parse_args()
    main(args.challenge_name, args.dataset) 