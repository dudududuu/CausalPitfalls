import os
import json
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Set, Union, Tuple
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


DEFAULT_LLM = 'all'
AVAILABLE_LLMS = ["gpt-4o", "gemini-2.0-flash", "claude-3-5-sonnet-20240620", "o4-mini-2025-04-16", "deepseek-chat"]

def load_rubric(challenge_name: str) -> Dict[str, Any]:
    """Load the rubric for a specific challenge."""
    rubric_path = Path(f"challenges/{challenge_name}/rubric.json")
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric not found at {rubric_path}")
    
    with open(rubric_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_answer_key(challenge_name: str) -> Dict[str, Any]:
    """Load the answer key for a specific challenge."""
    answer_key_path = Path(f"challenges/{challenge_name}/answer_key.json")
    if not answer_key_path.exists():
        raise FileNotFoundError(f"Answer key not found at {answer_key_path}")
    
    with open(answer_key_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def fill_rubric_template(rubric: Dict[str, Any], answer_key: Union[Dict[str, Any], List[Dict[str, Any]]], dataset_name: str) -> Dict[str, Any]:
    """Fill in the rubric template with values from the answer key"""
    
    if isinstance(answer_key, list):
        try:
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0]
            clean_dataset = str(dataset_name).replace('.csv', '')
            
            def clean_key_dataset(key):
                dataset = key.get('dataset', '')
                if isinstance(dataset, list):
                    dataset = dataset[0]
                return str(dataset).replace('.csv', '')
            
            matching_key = next(
                (key for key in answer_key if clean_dataset == clean_key_dataset(key)), 
                None
            )
            
            if not matching_key:
                matching_key = next(
                    (key for key in answer_key if clean_dataset in clean_key_dataset(key)), 
                    None
                )
            
            if matching_key:
                answer_key = matching_key
            else:
                answer_key = answer_key[0]
        except (IndexError, AttributeError) as e:
            raise ValueError(f"Invalid answer key format for dataset {dataset_name}")
    
    def process_value(value):
        try:
            if isinstance(value, str):
                result = value
                for key, replacement in answer_key.items():
                    try:
                        placeholders = [f"{{{key}}}", f"<{key}>"]
                        for placeholder in placeholders:
                            if placeholder in result:
                                if isinstance(replacement, list):
                                    replacement_str = f"({', '.join(str(item) for item in replacement)})"
                                else:
                                    replacement_str = str(replacement)
                                result = result.replace(placeholder, replacement_str)
                    except Exception as e:
                        continue
                return result
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            else:
                return value
        except Exception as e:
            raise
    
    try:
        return process_value(rubric)
    except Exception as e:
        raise

def get_result_files(challenge_name: str, dataset_name: str, llm_name: str) -> List[Tuple[Path, str]]:
    """Get all result files for the given challenge and dataset."""
    base_path = Path(f"challenges/{challenge_name}/results/{llm_name}/{dataset_name}/run")
    result_files = []
    
    # Walk through all difficulty levels
    for difficulty in ["very_easy", "easy", "medium", "hard", "very_hard"]:
        difficulty_path = base_path / difficulty
        if difficulty_path.exists():
            # Get both direct_analysis.txt and results_analysis.txt files
            direct_analysis = difficulty_path / "direct_analysis.txt"
            results_analysis = difficulty_path / "results_analysis.txt"
            
            if direct_analysis.exists():
                result_files.append((direct_analysis, "direct"))
            if results_analysis.exists():
                result_files.append((results_analysis, "results"))
    
    return result_files

def extract_response_from_file(file_path: Path) -> str:
    """Extract the response part from the analysis file."""
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16', 'utf-16le', 'utf-16be']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"Could not decode file {file_path} with any of the attempted encodings")
    
    # Find everything after "Response:"
    response_match = re.search(r'Response:(.*)', content, re.DOTALL)
    if response_match:
        return response_match.group(1).strip()
    return ""

def evaluate_with_gpt4o(rubric: Dict[str, Any], result_text: str) -> List[int]:
    """Evaluate results using GPT-4o and return binary scores."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Ensure all values in criteria are properly stringified
    def stringify_value(value):
        if isinstance(value, list):
            return [stringify_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: stringify_value(v) for k, v in value.items()}
        return str(value)
    
    # Create a copy of the criteria with stringified values
    criteria = stringify_value(rubric['criteria'])
    
    # Prepare the prompt for GPT-4o
    prompt = f"""You are an careful grader for causal reasoning answers. Evaluate the following text against these criteria. For each criterion, respond with 1 if the response satisfies the criterion, or 0 if it doesn't.

CRITERIA:
{json.dumps(criteria, indent=2)}

Response TO EVALUATE:
{result_text}

Return ONLY a JSON object with an array of binary scores (0 or 1) in the same order as the criteria. For example: {{"scores": [1,0,1,1,0,1,1,1]}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a critical evaluator for LLM generated answers. Provide only binary scores (0 or 1) for each criterion."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)["scores"]
    except Exception as e:
        print(f"Error parsing GPT-4o response: {str(e)}")
        print(f"Response content: {response.choices[0].message.content}")
        raise

def save_evaluation(scores: List[int], rubric: Dict[str, Any], challenge_name: str, dataset_name: str, 
                   difficulty: str, llm_name: str, protocol_type: str, response_text: str, file_path: Path):
    """Save the evaluation results in the results directory."""
    # Save in the same directory as the results
    output_dir = Path(f"challenges/{challenge_name}/results/{llm_name}/{dataset_name}/run/{difficulty}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create evaluation with full names
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
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    
    return evaluation

def save_summary_csv(all_evaluations: List[Dict[str, Any]], challenge_name: str, dataset_name: str, llm_name: str):
    """Save a summary CSV with all scores and responses, appending to existing data if present."""
    # Create DataFrame for new evaluations
    rows = []
    for eval_data in all_evaluations:
        meta = eval_data["metadata"]
        scores = eval_data["scores"]
        # Extract difficulty from the file path
        difficulty = Path(meta.get("file_path", "")).parent.name
        row = {
            "challenge": meta["challenge"],
            "dataset": meta["dataset"],
            "llm": meta["llm"],
            "protocol_type": meta["protocol_type"],
            "difficulty": difficulty,
            "total_score": eval_data["total_score"],
            "response": eval_data["response"]
        }
        # Add individual scores
        row.update(scores)
        rows.append(row)
    
    new_df = pd.DataFrame(rows)
    
    # Output directory and file
    output_dir = Path(f"challenges/{challenge_name}/results/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluations_summary.csv"
    
    # Check if the file already exists and is not empty
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            # Load existing data and concatenate with new data
            existing_df = pd.read_csv(output_path, encoding='utf-8')
            
            # Remove any existing rows for the current LLM to avoid duplicates
            if not existing_df.empty:
                existing_df = existing_df[existing_df["llm"] != llm_name]
            
            # Concatenate with new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Updated existing summary CSV with results for {llm_name}")
        except pd.errors.EmptyDataError:
            # File exists but is empty or corrupted
            new_df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Created new summary CSV with results for {llm_name} (previous file was empty or corrupted)")
    else:
        # Create new file if it doesn't exist or is empty
        new_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Created new summary CSV with results for {llm_name}")

def process_single_llm(llm_name: str, challenge: str, dataset: str, filled_rubric: Dict[str, Any]) -> None:
    """Process a single LLM's results."""
    print(f"\n{'='*50}")
    print(f"Processing LLM: {llm_name}")
    print(f"{'='*50}")
    
    # Get all result files
    result_files = get_result_files(challenge, dataset, llm_name)
    
    if not result_files:
        print(f"No result files found for dataset {dataset}")
        return
        
    print(f"Found {len(result_files)} result files to evaluate")
    
    # Store all evaluations for summary
    all_evaluations = []
    
    # Evaluate each result file
    for file_path, protocol_type in result_files:
        # Extract difficulty from the file path
        difficulty = file_path.parent.name
        
        print(f"\nEvaluating {difficulty}/{protocol_type}...")
        
        # Extract and evaluate the response
        result_text = extract_response_from_file(file_path)
        scores = evaluate_with_gpt4o(filled_rubric, result_text)
        
        # Save the evaluation and store for summary
        evaluation = save_evaluation(scores, filled_rubric, challenge, dataset, 
                                  difficulty, llm_name, protocol_type, result_text, file_path)
        all_evaluations.append(evaluation)
        
        print(f"Score: {sum(scores)}/{filled_rubric['max_score']}")
    
    # Save summary CSV
    if all_evaluations:
        save_summary_csv(all_evaluations, challenge, dataset, llm_name)
    else:
        print(f"No results found for {llm_name}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate results using GPT-4o and a rubric")
    parser.add_argument("--challenge", default=None, help="Name of the challenge (default: process all challenges)")
    parser.add_argument("--dataset", default=None, help="Name of the dataset (default: process all datasets)")
    parser.add_argument("--llm", default=DEFAULT_LLM, help=f"Name of the LLM (default: {DEFAULT_LLM}), or 'all' to process all available LLMs")
    args = parser.parse_args()

    try:
        # Get all challenge directories
        challenges_dir = Path("challenges")
        challenge_dirs = [d.name for d in challenges_dir.iterdir() if d.is_dir()]
        
        # If specific challenge is provided, only process that one
        if args.challenge:
            if args.challenge not in challenge_dirs:
                raise ValueError(f"Challenge '{args.challenge}' not found in challenges directory")
            challenge_dirs = [args.challenge]
            
        print(f"\nProcessing {len(challenge_dirs)} challenges: {', '.join(challenge_dirs)}")
            
        # Process each challenge
        for challenge_name in challenge_dirs:
            try:
                print(f"\n{'='*50}")
                print(f"Processing challenge: {challenge_name}")
                print(f"{'='*50}")
                
                # Load rubric and answer key for this challenge
                rubric = load_rubric(challenge_name)
                answer_key = load_answer_key(challenge_name)
                
                # Get all CSV files in the challenge directory
                challenge_path = challenges_dir / challenge_name
                datasets = [f.stem for f in challenge_path.glob("*.csv")]
                
                if not datasets:
                    print(f"Warning: No datasets found in challenge '{challenge_name}', skipping...")
                    continue
                
                print(f"\nFound datasets in {challenge_name}: {', '.join(datasets)}")
                    
                # If specific dataset is provided, only process that one
                if args.dataset:
                    if args.dataset not in datasets:
                        print(f"Warning: Dataset '{args.dataset}' not found in challenge '{challenge_name}', skipping...")
                        continue
                    datasets = [args.dataset]
                
                print(f"\nProcessing {len(datasets)} datasets in {challenge_name}: {', '.join(datasets)}")
                
                # Process each dataset
                for dataset_name in datasets:
                    try:
                        print(f"\n{'-'*40}")
                        print(f"Processing dataset: {dataset_name}")
                        print(f"{'-'*40}")
                        
                        # Fill rubric template with dataset-specific answer key
                        filled_rubric = fill_rubric_template(rubric, answer_key, dataset_name)
                        if args.llm.lower() == "all":
                            # Process all available LLMs
                            print(f"Processing all available LLMs: {', '.join(AVAILABLE_LLMS)}")
                            for llm_name in AVAILABLE_LLMS:
                                try:
                                    process_single_llm(llm_name, challenge_name, dataset_name, filled_rubric)
                                except Exception as e:
                                    print(f"Error processing LLM {llm_name} for dataset {dataset_name}: {str(e)}")
                                    continue
                        else:
                            # Process just the specified LLM
                            process_single_llm(args.llm, challenge_name, dataset_name, filled_rubric)
                        
                        print(f"\nSummary CSV saved to: challenges/{challenge_name}/results/{dataset_name}/evaluations_summary.csv")
                    except Exception as e:
                        print(f"Error processing dataset {dataset_name}: {str(e)}")
                        continue
                    
            except Exception as e:
                print(f"Error processing challenge {challenge_name}: {str(e)}")
                continue
        
        print("\nAll processing completed!")
        print("\nSummary of processing:")
        print(f"- Challenges processed: {len(challenge_dirs)}")
        print("- Results location: challenges/[challenge_name]/results/[dataset_name]/evaluations_summary.csv")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 