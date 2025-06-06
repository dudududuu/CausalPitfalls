import os
import pandas as pd
import glob


def ensure_results_dir(base_dir):
    """Create results directory if it doesn't exist"""
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def standardize_llm_name(name):
    """Standardize LLM names to match desired format"""
    name_map = {
        'claude-3-5-sonnet-20240620': 'Claude-3.5-sonnet',
        'gemini-2.0-flash': 'Gemini-2.0-flash',
        'gpt-4o': 'GPT-4o',
        'o4-mini-2025-04-16': 'GPT-o4-mini',
        'deepseek-chat': 'Deepseek-chat'
    }
    return name_map.get(name, name)

# Define global LLM order
LLM_ORDER = ['Claude-3.5-sonnet', 'Gemini-2.0-flash', 'Deepseek-chat', 'GPT-4o', 'GPT-o4-mini']

# Define the mapping of subchallenges to main pitfall categories
PITFALL_MAPPING = {
    'Confounding biases and spurious associations': {
        'short_name': 'Conf',
        'subchallenges': {
            'simpson_paradox': "Simpson's paradox",
            'berkson_paradox': "Selection bias (Berkson's paradox)"
        }
    },
    'Interventions and experimental reasoning': {
        'short_name': 'Interv',
        'subchallenges': {
            'observational_vs_experimental_reasoning': "Observational vs experimental",
            'casual_effect': "Causal effect estimation"
        }
    },
    'Counterfactual reasoning and hypotheticals': {
        'short_name': 'Counter',
        'subchallenges': {
            'counterfactual_reasoning': "Counterfactual prediction",
            'necessity_sufficiency': "Necessity and sufficiency"
        }
    },
    'Mediation and indirect causal effects': {
        'short_name': 'Med',
        'subchallenges': {
            'mediation_outcome_confounder': "Mediator-outcome confounding",
            'sequential_mediator': "Sequential mediators",
            'treatment_mediator': "Treatment-mediator interaction"
        }
    },
    'Causal discovery and structure learning': {
        'short_name': 'Disc',
        'subchallenges': {
            'causal_direction_iv': "Cause-effect direction",
            'dag_structure_markequi': "Structure uncertainty"
        }
    },
    'Causal generalization and external validity': {
        'short_name': 'Ext',
        'subchallenges': {
            'population_shift': "Population shift",
            'moderation_effect': "Contextual interaction and moderation",
            'temporal_stability': "Temporal stability",
            'domain_shift': "Domain adaptation"
        }
    }
}

def get_main_pitfall_category(subchallenge):
    """Map a subchallenge to its main pitfall category"""
    for category, info in PITFALL_MAPPING.items():
        if subchallenge in info['subchallenges']:
            return category
    return None

def get_short_name(category):
    """Get the short name for a main pitfall category"""
    return PITFALL_MAPPING[category]['short_name']

def get_challenge_description(subchallenge):
    """Get both the main category and subchallenge description"""
    for category, info in PITFALL_MAPPING.items():
        if subchallenge in info['subchallenges']:
            return {
                'main_category': category,
                'main_category_short': info['short_name'],
                'subchallenge_description': info['subchallenges'][subchallenge]
            }
    return None

def generate_performance_table(df):
    """
    Generate LaTeX table from aggregated results with separate sections for Direct and Code-Assisted prompting.
    Includes boldface for highest values in each column.
    """
    # Standardize difficulty levels before aggregating
    df = df.copy()
    df['difficulty'] = df['difficulty'].str.lower()
    
    # Standardize LLM names
    df['llm'] = df['llm'].apply(standardize_llm_name)
    
    # Map subchallenges to main categories
    df['main_category'] = df['challenge'].apply(get_main_pitfall_category)
    
    # Group by llm, protocol_type, main_category and take mean normalized_score
    grouped = df.groupby(['llm', 'protocol_type', 'main_category'])['normalized_score'].mean().reset_index()
    
    # Create pivot tables for each protocol type
    pivot_direct = grouped[grouped['protocol_type'] == 'direct'].pivot_table(
        index='llm',
        columns='main_category',
        values='normalized_score',
        aggfunc='mean'
    ).round(2)
    
    pivot_code = grouped[grouped['protocol_type'] == 'results'].pivot_table(
        index='llm',
        columns='main_category',
        values='normalized_score',
        aggfunc='mean'
    ).round(2)
    
    # Calculate aggregate scores
    agg_direct = pivot_direct.mean(axis=1).round(2)
    agg_code = pivot_code.mean(axis=1).round(2)
    
    # Generate LaTeX table
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\caption{Causal reliability across causal pitfalls, comparing direct and code-assisted prompting. Values represent averages of normalized scores, defined in equation~\eqref{eq:normalized_score}, across five questions per pitfall category; higher scores indicate better performance.}")
    latex.append(r"\label{tab:overall_performance}")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\begin{tabular}{lccccccc}")
    latex.append(r"\toprule")
    
    # Column headers using short names
    headers = [get_short_name(cat) for cat in PITFALL_MAPPING.keys()]
    latex.append(r"\textbf{LLM (Direct Prompting)} & " + " & ".join(headers) + r" & Average \\")
    latex.append(r"\midrule")
    
    def format_value(value, is_max):
        if pd.isna(value):
            return "-"
        formatted = f"{value:.2f}"
        return rf"\textbf{{{formatted}}}" if is_max else formatted
    
    # Add rows for each protocol type
    for section_data, pivot_table, agg_scores in [(pivot_direct, pivot_direct, agg_direct), (pivot_code, pivot_code, agg_code)]:
        if section_data is pivot_code:
            latex.append(r"\bottomrule")
            latex.append(r"\\[8pt]")
            latex.append(r"\toprule")
            latex.append(r"\textbf{LLM (Code-Assisted Prompting)} & " + " & ".join(headers) + r" & Average \\")
            latex.append(r"\midrule")
        
        max_vals = pivot_table.max()
        max_agg = agg_scores.max()
        
        for llm in LLM_ORDER:
            if llm not in pivot_table.index:
                continue
            row = [llm]
            for category in PITFALL_MAPPING.keys():
                try:
                    value = pivot_table.loc[llm, category]
                    is_max = (value == max_vals[category])
                    row.append(format_value(value, is_max))
                except:
                    row.append("-")
            avg_value = agg_scores[llm]
            is_max_avg = (avg_value == max_agg)
            row.append(format_value(avg_value, is_max_avg))
            latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\vspace{2pt}")
    latex.append(r"\begin{flushleft}")
    latex.append(r"{\footnotesize\textit{")
    descriptions = []
    for category, info in PITFALL_MAPPING.items():
        descriptions.append(f"{info['short_name']}: {category}")
    latex.append("; ".join(descriptions))
    latex.append(r"}}")
    latex.append(r"\end{flushleft}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def generate_difficulty_performance_table(df):
    """
    Generate LaTeX table showing performance by difficulty level for both direct and code-assisted prompting.
    """
    # Standardize difficulty levels
    df = df.copy()
    df['difficulty'] = df['difficulty'].str.lower()
    
    # Define difficulty mappings and order
    difficulty_map = {
        'very_easy': 'Very Easy',
        'easy': 'Easy', 
        'medium': 'Medium',
        'hard': 'Hard',
        'very_hard': 'Very Hard'
    }
    difficulty_order = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    
    # Map difficulties
    df['difficulty'] = df['difficulty'].map(difficulty_map)
    
    # Group by llm, protocol_type, difficulty and take mean normalized_score
    grouped = df.groupby(['llm', 'protocol_type', 'difficulty'])['normalized_score'].mean().reset_index()
    
    # Create pivot tables for each protocol type
    pivot_direct = grouped[grouped['protocol_type'] == 'direct'].pivot_table(
        index='llm',
        columns='difficulty',
        values='normalized_score',
        aggfunc='mean'
    ).round(2)
    
    pivot_code = grouped[grouped['protocol_type'] == 'results'].pivot_table(
        index='llm',
        columns='difficulty',
        values='normalized_score',
        aggfunc='mean'
    ).round(2)
    
    # Generate LaTeX table
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\caption{Causal reliability by difficulty levels of questions, comparing direct and code-assisted prompting. Values represent averages of normalized score, defined in equation~\eqref{eq:normalized_score}, across 16 challenges; higher scores indicate better performance.}")
    latex.append(r"\label{tab:difficulty_performance}")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\toprule")
    
    # Column headers
    headers = difficulty_order
    latex.append(r"\textbf{LLM (Direct Prompting)} & " + " & ".join(headers) + r" \\")
    latex.append(r"\midrule")
    
    def format_value(value):
        if pd.isna(value):
            return "-"
        return f"{value:.2f}"
    
    # Add rows for each protocol type
    for section_data, pivot_table in [(pivot_direct, pivot_direct), (pivot_code, pivot_code)]:
        if section_data is pivot_code:
            latex.append(r"\bottomrule")
            latex.append(r"\\[8pt]")
            latex.append(r"\toprule")
            latex.append(r"\textbf{LLM (Code-Assisted Prompting)} & " + " & ".join(headers) + r" \\")
            latex.append(r"\midrule")
        
        for llm in LLM_ORDER:
            if llm not in pivot_table.index:
                continue
            row = [llm]
            for diff in difficulty_order:
                try:
                    value = pivot_table.loc[llm, diff]
                    row.append(format_value(value))
                except:
                    row.append("-")
            latex.append(" & ".join(row) + r" \\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def read_csv_with_encoding(file_path):
    """Try reading CSV with different encodings"""
    encodings = ['utf-8', 'cp1252', 'latin1', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with encoding detection
    try:
        import chardet
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        detected = chardet.detect(raw_data)
        return pd.read_csv(file_path, encoding=detected['encoding'])
    except Exception as e:
        raise ValueError(f"Failed to read file with any encoding: {str(e)}")

def calculate_error_rates(error_df, results_dir):
    """Calculate error rates by LLM and main category, then by difficulty"""
    # Standardize difficulty levels
    error_df = error_df.copy()
    error_df['difficulty'] = error_df['difficulty'].str.lower()
    
    # Drop rows with missing main_category
    before = len(error_df)
    error_df = error_df.dropna(subset=['main_category'])
    after = len(error_df)
    if after < before:
        print(f"Warning: Dropped {before - after} rows with missing main_category.")
    
    # First calculate error rates by LLM and main category
    category_rates = error_df.groupby(['llm', 'main_category'])['has_error'].agg(['mean', 'count']).reset_index()
    category_rates['error_rate'] = (category_rates['mean'] * 100).round(2)
    category_rates['total_attempts'] = category_rates['count']
    
    # Sort by error rate to see which categories have highest errors
    category_rates = category_rates.sort_values(['llm', 'error_rate'], ascending=[True, False])
    
    # Save category-based results
    category_output = os.path.join(results_dir, 'error_rates_by_category.csv')
    category_rates[['llm', 'main_category', 'error_rate', 'total_attempts']].to_csv(category_output, index=False)
    print(f"Error rates by category saved to: {category_output}")
    
    # Print top error categories for each LLM
    print("\nTop Error Categories by LLM:")
    for llm in LLM_ORDER:
        llm_data = category_rates[category_rates['llm'] == llm].head(3)
        if not llm_data.empty:
            print(f"\n{llm}:")
            print(llm_data[['main_category', 'error_rate', 'total_attempts']].to_string(index=False))
    
    # Then calculate error rates by difficulty and LLM
    difficulty_rates = error_df.groupby(['difficulty', 'llm'])['has_error'].mean().reset_index()
    difficulty_rates['error_rate'] = (difficulty_rates['has_error'] * 100).round(2)
    
    # Sort by difficulty and LLM
    difficulty_rates = difficulty_rates.sort_values(['difficulty', 'llm'])
    
    # Save difficulty-based results
    difficulty_output = os.path.join(results_dir, 'error_rates_by_difficulty.csv')
    difficulty_rates[['difficulty', 'llm', 'error_rate']].to_csv(difficulty_output, index=False)
    print(f"\nError rates by difficulty saved to: {difficulty_output}")
    
    # Print summary by difficulty
    print("\nError Rates by Difficulty and LLM:")
    print(difficulty_rates[['difficulty', 'llm', 'error_rate']].to_string(index=False))

def generate_branding_bias_table(csv_path, results_dir):
    """
    Generate LaTeX table for branding bias results.
    """
    # Read the branding bias CSV file
    df = pd.read_csv(csv_path)
    
    # Create a dictionary to store results by scenario and model
    results = {}
    
    # Process each row
    for _, row in df.iterrows():
        ground_truth = row['Ground Truth']
        product = row['Product']
        model = standardize_llm_name(row['Model'])
        conclusion = row['LLM Conclusion']
        
        # Create scenario key
        scenario_key = f"{ground_truth}_{product}"
        
        if scenario_key not in results:
            results[scenario_key] = {
                'ground_truth': ground_truth,
                'product': product,
                'models': {}
            }
        
        # Store model result without checkmark
        results[scenario_key]['models'][model] = conclusion
    
    # Generate LaTeX table
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\caption{Branding bias results across different models and scenarios.}")
    latex.append(r"\label{tab:branding_bias_results}")
    latex.append(r"\small")
    latex.append(r"\renewcommand{\arraystretch}{1.3}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Ground Truth Effect} & \textbf{Product} & \textbf{GPT-4o} & \textbf{Claude-3} & \textbf{Gemini-2} \\")
    latex.append(r"\midrule")
    
    # Add rows for each scenario
    for scenario_key, data in results.items():
        row = [
            data['ground_truth'],
            data['product'],
            data['models'].get('GPT-4o', '-'),
            data['models'].get('Claude-3.5-sonnet', '-'),
            data['models'].get('Gemini-2.0-flash', '-')
        ]
        latex.append(" & ".join(row) + r" \\[3pt]")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    # Save the LaTeX table
    output_path = os.path.join(results_dir, 'branding_bias_table.tex')
    with open(output_path, 'w') as f:
        f.write("\n".join(latex))
    print(f"Branding bias LaTeX table saved to: {output_path}")

def merge_evaluation_csvs():
    """
    Merge all evaluations_summary.csv files from challenges directory into one large csv.
    Keep only essential columns, add max_score column, and calculate normalized scores.
    Generate LaTeX table with aggregated results and track error rates.
    """
    # Find all evaluations_summary.csv files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(base_dir, 'challenges', '*', 'results', '*', 'evaluations_summary.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError("No evaluations_summary.csv files found")

    # Create results directory
    results_dir = ensure_results_dir(base_dir)

    # Generate branding bias table if the file exists
    branding_bias_csv = os.path.join(base_dir, 'ablations', 'branding_bias', 'results', 'branding_bias_all_models_summary.csv')
    if os.path.exists(branding_bias_csv):
        generate_branding_bias_table(branding_bias_csv, results_dir)

    # List to store all dataframes
    dfs = []
    error_dfs = []
    unmapped_challenges = set()
    
    for file_path in csv_files:
        try:
            # Read CSV with proper encoding handling
            df = read_csv_with_encoding(file_path)
            
            # Get scoring columns
            response_idx = df.columns.get_loc('response')
            scoring_cols = df.columns[response_idx + 1:].tolist()
            
            # Calculate max possible score
            max_score = len(scoring_cols)
            
            # Create error tracking dataframe with proper copy
            error_cols = ['challenge', 'dataset', 'llm', 'protocol_type', 'difficulty', 'response']
            error_df = df[error_cols].copy()
            error_df.loc[:, 'has_error'] = df['response'].str.contains('error|exception|failed|traceback', case=False, na=False).astype(int)
            
            # Add main_category column with logging for unmapped challenges
            def get_main_category_with_logging(challenge):
                desc = get_challenge_description(challenge)
                if not desc:
                    unmapped_challenges.add(challenge)
                    return None
                return desc['main_category']
            
            error_df['main_category'] = error_df['challenge'].apply(get_main_category_with_logging)
            error_dfs.append(error_df.drop('response', axis=1))
            
            # Keep essential columns with proper copy
            essential_cols = ['challenge', 'dataset', 'llm', 'protocol_type', 'difficulty', 'total_score']
            df_essential = df[essential_cols].copy()
            
            # Add max_score and calculate normalized score
            df_essential.loc[:, 'max_score'] = max_score
            df_essential.loc[:, 'normalized_score'] = (df_essential['total_score'] / df_essential['max_score']) * 100
            
            dfs.append(df_essential)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            print(f"Skipping this file and continuing with others...")
            continue
    
    if not dfs:
        raise ValueError("No data was successfully loaded from any CSV file")
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    error_df = pd.concat(error_dfs, ignore_index=True)
    
    # Print warning about unmapped challenges
    if unmapped_challenges:
        print("\nWARNING: Found challenges that are not mapped in PITFALL_MAPPING:")
        for challenge in sorted(unmapped_challenges):
            print(f"  - {challenge}")
        print("\nPlease add these challenges to PITFALL_MAPPING in get_final_results.py")
    
    # Calculate error rates
    calculate_error_rates(error_df, results_dir)
    
    # Add challenge category information with the same logging
    merged_df['main_category'] = merged_df['challenge'].apply(get_main_category_with_logging)
    merged_df['main_category_short'] = merged_df['challenge'].apply(lambda x: get_challenge_description(x)['main_category_short'] if get_challenge_description(x) else None)
    merged_df['subchallenge_description'] = merged_df['challenge'].apply(lambda x: get_challenge_description(x)['subchallenge_description'] if get_challenge_description(x) else None)
    
    # Create aggregated statistics with both levels
    agg_stats = (merged_df.groupby(['llm', 'main_category', 'main_category_short', 'protocol_type'])['normalized_score']
                 .agg(['mean', 'count', 'std'])
                 .round(2)
                 .reset_index())
    
    # Rename columns for clarity
    agg_stats.columns = ['llm', 'main_category', 'main_category_short', 'protocol_type', 'normalized_score_mean', 'n_subchallenges', 'normalized_score_std']
    
    # Generate and save LaTeX tables
    latex_table = generate_performance_table(merged_df)
    latex_output_path = os.path.join(results_dir, 'performance_table.tex')
    with open(latex_output_path, 'w') as f:
        f.write(latex_table)
    print(f"Performance LaTeX table saved to: {latex_output_path}")
    
    # Generate and save difficulty-based LaTeX table
    difficulty_latex_table = generate_difficulty_performance_table(merged_df)
    difficulty_latex_output_path = os.path.join(results_dir, 'difficulty_performance_table.tex')
    with open(difficulty_latex_output_path, 'w') as f:
        f.write(difficulty_latex_table)
    print(f"Difficulty-based LaTeX table saved to: {difficulty_latex_output_path}")
    
    # Save detailed merged dataframe
    output_path = os.path.join(results_dir, 'overall_evaluations.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"Merged evaluations saved to: {output_path}")
    
    # Save aggregated statistics
    agg_output_path = os.path.join(results_dir, 'aggregated_results.csv')
    agg_stats.to_csv(agg_output_path, index=False)
    print(f"Aggregated results saved to: {agg_output_path}")

if __name__ == "__main__":
    merge_evaluation_csvs() 