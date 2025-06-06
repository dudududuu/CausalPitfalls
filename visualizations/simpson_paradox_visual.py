import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('visualizations/figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

np.random.seed(42)

# Generate synthetic data
n_samples = 300

# Create two groups with clear separation
group = np.random.choice(['Group A', 'Group B'], size=n_samples, p=[0.6, 0.4])

# Generate treatment consumption (continuous)
# Make treatment levels distinct between groups
treatment = []
for g in group:
    if g == 'Group A':
        treatment.append(np.random.uniform(6, 10))  # Higher treatment for Group A
    else:
        treatment.append(np.random.uniform(0, 4))   # Lower treatment for Group B
treatment = np.array(treatment)

# Generate outcome with Simpson's paradox
outcome = []
for g, t in zip(group, treatment):
    if g == 'Group A':
        # Group A: strong negative effect of treatment
        base_outcome = 90 - 3 * t + np.random.normal(0, 3)
    else:
        # Group B: strong negative effect of treatment
        base_outcome = 60 - 4 * t + np.random.normal(0, 3)
    outcome.append(base_outcome)

# Create DataFrame
data = pd.DataFrame({
    'Group': group,
    'Treatment': treatment,
    'Outcome': outcome
})

# Figure 1: Subgroup fits
plt.figure(figsize=(4, 3))
sns.scatterplot(data=data, x='Treatment', y='Outcome', hue='Group', 
                marker='s', s=30, alpha=0.6)  # Using square markers

# Add regression lines for subgroups
sns.regplot(data=data[data['Group'] == 'Group A'], 
            x='Treatment', y='Outcome', scatter=False, 
            line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5})
sns.regplot(data=data[data['Group'] == 'Group B'], 
            x='Treatment', y='Outcome', scatter=False, 
            line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5})

plt.xlabel('Treatment Consumption', fontsize=10, labelpad=10)
plt.ylabel('Health Outcome', fontsize=10, labelpad=10)
plt.title("Subgroup Analysis", fontsize=12, pad=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(title='', frameon=False, loc='upper right', fontsize=9)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/figures/simpson_paradox_subgroups.pdf', dpi=300, bbox_inches='tight', transparent=True)

# Figure 2: Overall fit
plt.figure(figsize=(4, 3))
sns.scatterplot(data=data, x='Treatment', y='Outcome', 
                marker='s', s=30, alpha=0.6, color='gray')  # Using gray squares

# Add overall regression line
sns.regplot(data=data, x='Treatment', y='Outcome', 
            scatter=False, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5})

plt.xlabel('Treatment Consumption', fontsize=10, labelpad=10)
plt.ylabel('Health Outcome', fontsize=10, labelpad=10)
plt.title("Overall Analysis", fontsize=12, pad=15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(axis='both', which='major', labelsize=9)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/figures/simpson_paradox_overall.pdf', dpi=300, bbox_inches='tight', transparent=True)
