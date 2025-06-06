import numpy as np
import pandas as pd
import os
import statsmodels.api as sm

np.random.seed(42)

def generate_confounder_data(n_samples=500):
    """
    Generate data for the confounder scenario:
    High temperature -> More ice cream sales
    High temperature -> More drowning incidents
    Here, temperature is a confounder that affects both ice cream sales and drownings
    """
    # Generate temperature (confounder)
    temperature = np.random.normal(loc=75, scale=15, size=n_samples)
    
    # Generate ice cream sales based on temperature with some noise
    # Higher temperature leads to more ice cream sales
    ice_cream_sales = 10 + 0.5 * temperature + np.random.normal(scale=10, size=n_samples)
    
    drowning_incidents =  0.5*temperature + np.random.normal(scale=0.5, size=n_samples)
    
    # Make sure drowning incidents are non-negative
    drowning_incidents = np.maximum(drowning_incidents, 0)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'ice_cream_sales': ice_cream_sales,
        'drowning_incidents': drowning_incidents
    })
    
    return df

def generate_pure_noise_confounder_data(n_samples=100):
    """
    Generate pure_noise/random data for the confounder scenario with no true causal relationship.
    All variables are randomly generated independently to ensure near-zero correlations.
    """
    
    temp_rng = np.random.RandomState(42)
    ice_rng = np.random.RandomState(99)
    drown_rng = np.random.RandomState(456)
    
    # Generate variables
    temperature = temp_rng.normal(loc=75, scale=10, size=n_samples)
    ice_cream_sales = ice_rng.normal(loc=50, scale=10, size=n_samples)
    drowning_incidents = drown_rng.normal(loc=2, scale=0.8, size=n_samples)
    
    # Make sure drowning incidents are non-negative
    drowning_incidents = np.maximum(drowning_incidents, 0)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'temperature': temperature,
        'ice_cream_sales': ice_cream_sales,
        'drowning_incidents': drowning_incidents
    })
    
    return df

def main():
    """Generate and save the datasets"""
    # Create output directory if it doesn't exist
    os.makedirs('ablations/domain_knowledge/data', exist_ok=True)
    
    # Generate confounder data
    confounder_data = generate_confounder_data()
    
    # Define column mappings for confounder scenario
    confounder_mapping = {
        'temperature': 'X1',
        'ice_cream_sales': 'X2',
        'drowning_incidents': 'Y'
    }

    # Generate pure_noise confounder data with the same column names
    pure_noise_confounder = generate_pure_noise_confounder_data()


    # Save all versions
    confounder_data.to_csv('ablations/domain_knowledge/data/confounder_data.csv', index=False)
    pure_noise_confounder.to_csv('ablations/domain_knowledge/data/confounder_data_pure_noise.csv', index=False)
    
    print(f"Confounder data generated with {len(confounder_data)} samples")
    print(f"pure_noise confounder data generated with {len(pure_noise_confounder)} samples")

    
    
    # Print correlation matrices
    print("\nCorrelation Matrix for Confounder Data:")
    print(confounder_data.corr().round(2))
    
    print("\nCorrelation Matrix for pure_noise Confounder Data:")
    print(pure_noise_confounder.corr().round(2))
    
    # Run linear regression analysis for confounder data
    print("\n----- Linear Regression Analysis -----")
    print("\nConfounder Scenario (drowning ~ temp + ice_cream):")
    X = confounder_data[['temperature', 'ice_cream_sales']]
    X = sm.add_constant(X)
    y = confounder_data['drowning_incidents']
    model = sm.OLS(y, X).fit()
    print(model.summary().tables[1])
    print(f"P-value for temperature: {model.pvalues.iloc[1]:.6f} {'(significant)' if model.pvalues.iloc[1] < 0.05 else '(not significant)'}")
    # Run linear regression analysis for pure_noise confounder data
    print("\n----- Linear Regression Analysis -----")
    print("\nNonsensical Confounder Scenario (drowning ~ temp + ice_cream):")
    X = pure_noise_confounder[['temperature', 'ice_cream_sales']]
    X = sm.add_constant(X)
    y = pure_noise_confounder['drowning_incidents']
    model = sm.OLS(y, X).fit()
    print(model.summary().tables[1])
    print(f"P-value for temperature: {model.pvalues.iloc[1]:.6f} {'(significant)' if model.pvalues.iloc[1] < 0.05 else '(not significant)'}")
    

if __name__ == "__main__":
    main() 