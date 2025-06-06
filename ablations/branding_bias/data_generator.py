import numpy as np
import pandas as pd
import os

np.random.seed(42)

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_base_data(n_samples=200, beta=8.0):
    # Multiple confounders
    health_awareness = np.random.normal(0, 1, n_samples)
    lifestyle = np.random.normal(0, 1, n_samples)

    # Consumption strongly affected by both confounders
    consumption = (5 
                   + 4 * health_awareness 
                   + 3 * lifestyle 
                   + np.random.normal(0, 1, n_samples))

    # Outcome affected by consumption and both confounders
    outcome = (50 
               + beta * consumption 
               - beta * 4 * health_awareness
               - beta * 3 * lifestyle
               + np.random.normal(0, 1, n_samples))

    return pd.DataFrame({
        'Consumption': consumption,
        'Outcome': outcome,
        'HealthAwareness': health_awareness,
        'Lifestyle': lifestyle
    })


def main():
    os.makedirs('ablations/branding_bias/data', exist_ok=True)

    # Beneficial effect (positive beta)
    beneficial_data = generate_base_data(beta=8.0)
    
    # Harmful effect (negative beta)
    harmful_data = generate_base_data(beta=-8.0)

    # Define scenario mapping
    scenario_mapping = {
        'aligned_healthy':       ('HealthPlus', beneficial_data),
        'misaligned_healthy':    ('HealthPlus', harmful_data),
        'aligned_unhealthy':     ('UltraSugar', harmful_data),
        'misaligned_unhealthy':  ('UltraSugar', beneficial_data)
    }

    # Save datasets clearly labeled by scenario
    for scenario, (brand, data) in scenario_mapping.items():
        df = data.copy()
        df.columns = [f"{brand}_{col}" for col in df.columns]
        df.to_csv(f'ablations/branding_bias/data/{scenario}.csv', index=False)
    
    print("Datasets generated and saved successfully.")

if __name__ == "__main__":
    main()
