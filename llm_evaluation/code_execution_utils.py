import os
import re
import logging
import subprocess
import codecs
from typing import Dict, Set, Optional, Tuple
import tempfile

def save_error_info(question_dir: str, error_msg: str) -> None:
    """Save error information to a file in the question directory."""
    with codecs.open(os.path.join(question_dir, 'error.txt'), 'w', encoding='utf-8') as f:
        f.write(error_msg)

def save_execution_results(question_dir: str, results: str) -> None:
    """Save execution results to a file in the question directory."""
    with codecs.open(os.path.join(question_dir, 'execution_results.txt'), 'w', encoding='utf-8') as f:
        f.write(results)

def execute_code_with_dependencies(
    extracted_code: str,
    question_dir: str,
    venv_dir: Optional[str] = None
) -> str:
    """
    Execute Python code with proper dependency management and error handling.
    
    Args:
        extracted_code (str): The Python code to execute
        question_dir (str): Directory where temporary files and logs will be stored
        venv_dir (Optional[str]): Path to virtual environment. If None, uses '.venv' in current directory
        
    Returns:
        str: Execution results or error message
    """
    # Set up virtual environment paths
    if venv_dir is None:
        venv_dir = os.path.abspath('.venv')
    
    if not os.path.exists(venv_dir):
        error_msg = f"Virtual environment not found at {venv_dir}"
        logging.error(error_msg)
        save_error_info(question_dir, error_msg)
        return "NULL"
    
    venv_python = os.path.join(venv_dir, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'python')
    if not os.path.exists(venv_python):
        error_msg = f"Python interpreter not found at {venv_python}"
        logging.error(error_msg)
        save_error_info(question_dir, error_msg)
        return "NULL"

    # Define package configurations
    causal_packages = {
        'statsmodels': 'statsmodels',
        'linearmodels': 'linearmodels>=4.0',
        'causalnex': 'causalnex',
        'dowhy': 'dowhy',
        'causalml': 'causalml',
        'econml': 'econml',
        'causalimpact': 'causalimpact',
        'causalinference': 'causalinference',
        'pycausal': 'pycausal',
    }
    
    package_map = {
        'np': 'numpy',
        'pd': 'pandas',
        'sm': 'statsmodels',
        'plt': 'matplotlib',
        'sns': 'seaborn',
        'sklearn': 'scikit-learn',
        'torch': 'pytorch',
        'tf': 'tensorflow',
        'xgb': 'xgboost',
        'econml': 'econml',
        'dowhy': 'dowhy', 
        'causalml': 'causalml',
        'linearmodels': 'linearmodels',
        'causalnex': 'causalnex',
        'statsmodels': 'statsmodels',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'causalimpact': 'causalimpact',
        'cvxpy': 'cvxpy',
        'networkx': 'networkx',
        'sympy': 'sympy',
        'pymc': 'pymc',
        'math': None,
        'os': None,
        'sys': None,
        'datetime': None,
        'json': None,
        're': None,
        'time': None,
        'warnings': None,
        'logging': None
    }
    
    problematic_packages = {
        'econml': "econml has complex dependencies including tensorflow and requires specific versions of packages.",
        'pycausal': "pycausal requires Java and specific configurations."
    }
    
    known_submodules = {
        'logit', 'ols', 'glm', 'formula', 'api',  # statsmodels
        'pyplot', 'figure', 'axes',  # matplotlib
        'io', 'core', 'util',  # pandas
        'random', 'linalg', 'fft',  # numpy
        'metrics', 'model_selection', 'preprocessing',  # sklearn
        'dml', 'metalearners', 'inference',  # econml
        'causal', 'graph', 'explain',  # dowhy
        'propensity', 'matching', 'effect',  # various causal packages
        'iv', 'panel', 'discrete', 'system'  # linearmodels
    }

    # Standard imports and helper functions
    standard_imports = """
# Standard imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, logit
import matplotlib.pyplot as plt
import seaborn as sns
import os  

# Helper function for safe path handling
def get_safe_path(*parts):
    return os.path.join(*[str(part) for part in parts]).replace('\\\\', '/')
"""

    missing_package_handler = """
# Handle missing packages gracefully
def check_import(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"Warning: Package {package_name} is not installed. Some functionality will be limited.")
        return False

has_econml = check_import('econml')
has_dowhy = check_import('dowhy')
has_causalml = check_import('causalml')
has_linearmodels = check_import('linearmodels')
has_causalnex = check_import('causalnex')
"""

    try:
        # Install common causal packages
        with codecs.open(os.path.join(question_dir, 'package_installation.log'), 'w', encoding='utf-8') as log_file:
            log_file.write("Installing common causal inference packages...\n")
            
            for package_name, install_spec in causal_packages.items():
                if package_name in problematic_packages:
                    log_file.write(f"Skipping {package_name}: {problematic_packages[package_name]}\n")
                    continue
                    
                try:
                    try:
                        subprocess.run([venv_python, '-c', f'import {package_name}'], 
                                     capture_output=True, 
                                     check=True)
                    except subprocess.CalledProcessError:
                        result = subprocess.run(
                            [venv_python, '-m', 'pip', 'install', install_spec, '--upgrade'], 
                            capture_output=True, 
                            text=True
                        )
                except Exception as e:
                    log_file.write(f"Error during {package_name} installation: {str(e)}\n")

        # Install packages from the code
        imports = re.findall(r'import\s+(\w+)|from\s+(\w+)(?:\.\w+)*\s+import', extracted_code)
        packages = set([imp[0] or imp[1] for imp in imports if imp[0] or imp[1]]) if imports else set()
        
        for package in packages:
            if package in known_submodules:
                continue
                
            package_to_install = package_map.get(package, package)
            
            if package_to_install is None or package_to_install in problematic_packages:
                continue
            
            try:
                check_result = subprocess.run(
                    [venv_python, '-c', f'import {package}'], 
                    capture_output=True, 
                    text=True,
                    check=False
                )
                
                if check_result.returncode != 0:
                    install_result = subprocess.run(
                        [venv_python, '-m', 'pip', 'install', package_to_install], 
                        capture_output=True, 
                        text=True,
                        check=False
                    )
                    
                    if install_result.returncode != 0:
                        warning_msg = f"Failed to install {package_to_install}: {install_result.stderr}"
                        with codecs.open(os.path.join(question_dir, 'warning.txt'), 'a', encoding='utf-8') as f:
                            f.write(f"{warning_msg}\n")
            except Exception as e:
                warning_msg = f"Error checking/installing package {package}: {str(e)}"
                with codecs.open(os.path.join(question_dir, 'warning.txt'), 'a', encoding='utf-8') as f:
                    f.write(f"{warning_msg}\n")

        # Update code to handle file paths safely
        if 'pd.read_csv' in extracted_code:
            extracted_code = re.sub(
                r"pd\.read_csv\(['\"]([^'\"]+)['\"]\)", 
                lambda m: f"pd.read_csv(get_safe_path({', '.join(repr(p) for p in m.group(1).split('/'))}))".replace('\\', '/'),
                extracted_code
            )

        # Create and execute temporary file
        temp_file = os.path.join(question_dir, 'temp_analysis.py')
        with codecs.open(temp_file, 'w', encoding='utf-8') as f:
            f.write(standard_imports + "\n" + missing_package_handler + "\n" + extracted_code)
        
        process = subprocess.run(
            [venv_python, temp_file], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        if process.returncode != 0:
            error_details = process.stderr if process.stderr else "No error details available"
            error_msg = f"Code execution failed with return code {process.returncode}\nError: {error_details}"
            logging.error(error_msg)
            save_error_info(question_dir, error_msg)
            return f"ERROR: {error_details}"
        else:
            execution_results = process.stdout
            if not execution_results.strip():
                warning_msg = "Code executed successfully but produced no output"
                logging.warning(warning_msg)
                with codecs.open(os.path.join(question_dir, 'warning.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"Warning: {warning_msg}\n\nCode:\n{extracted_code}")
                return "NOTE: Code executed successfully but produced no output"
            
            save_execution_results(question_dir, execution_results)
            return execution_results

    except Exception as e:
        error_msg = f"Error during code execution: {str(e)}"
        logging.error(error_msg)
        save_error_info(question_dir, error_msg)
        return "NULL"

# Set up virtual environment and package management
venv_path = os.path.join(os.path.dirname(__file__), '.venv')
python_path = os.path.join(venv_path, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_path, 'bin', 'python')

# Define allowed packages and their configurations
ALLOWED_PACKAGES = {
    'numpy': {'imports': ['numpy as np']},
    'pandas': {'imports': ['pandas as pd']},
    'matplotlib': {'imports': ['matplotlib.pyplot as plt']},
    'statsmodels': {'imports': ['statsmodels.api as sm']},
    'scikit-learn': {'imports': ['sklearn']},
    'econml': {'imports': ['econml']},
    'dowhy': {'imports': ['dowhy']},
    'linearmodels': {'imports': ['linearmodels']}
}

# Common imports and helper functions
def safe_path_join(*paths):
    """Safely join paths using forward slashes."""
    return '/'.join(str(p).replace('\\', '/') for p in paths)

def handle_missing_package(package_name):
    """Handle missing package gracefully with informative error."""
    return f"Error: Required package '{package_name}' not found. Please install it using pip."

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([python_path, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def execute_code(code, timeout=30):
    """Execute code in a safe environment with proper error handling."""
    try:
        # Create temporary file with proper encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name

        # Execute code with timeout
        result = subprocess.run(
            [python_path, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Clean up temporary file
        os.unlink(temp_path)

        if result.returncode == 0:
            return result.stdout.strip() or "Code executed successfully but produced no output"
        else:
            return f"Error: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Error: {str(e)}" 