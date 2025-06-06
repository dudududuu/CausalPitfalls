import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

plt.rcParams['font.family']      = 'Arial'
plt.rcParams['font.size']        = 8
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# Ensure output directory exists
os.makedirs('visualizations/figures', exist_ok=True)

# Load and prepare the results data
df = pd.read_csv('results/aggregated_results.csv')

# Shorten names
challenge_map = {
    'Confounding biases and spurious associations': 'Confounding',
    'Interventions and experimental reasoning': 'Experimental Reasoning',
    'Counterfactual reasoning and hypotheticals': 'Counterfactual Reasoning',
    'Mediation and indirect causal effects': 'Mediation Effects',
    'Causal discovery and structure learning': 'Causal Discovery',
    'Causal generalization and external validity': 'External Validity'
}

# Filter & rename
df = df[df['main_category'].isin(challenge_map)]
df['challenge'] = df['main_category'].map(challenge_map)

# Wrapped labels
wrapped_labels = [
    'Confounding',
    'Experimental\nReasoning',
    'Counterfactual\nReasoning',
    'Mediation\nEffects',
    'Causal\nDiscovery',
    'External\nValidity'
]

def create_radar_plot(data, title, ax):
    categories = list(challenge_map.values())
    N = len(categories)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  

    color_list = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Tick positions & wrapped labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(wrapped_labels, fontsize=8, fontweight='medium')
    ax.xaxis.set_tick_params(pad=12)   # push labels outward

    # Rotate & align each label tangentially
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        deg = np.degrees(angle)
        # align left on the right-side, right on the left-side
        ha = 'left'  if 0 < deg < 180 else 'right'
        va = 'center'
        rot = deg - 90 if deg <= 180 else deg + 90
        label.set_horizontalalignment(ha)
        label.set_verticalalignment(va)
        label.set_rotation(rot)

    # Radius settings
    ax.set_rscale('linear')
    ax.set_ylim(0, 100)
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels(['20','40','60','80'], fontsize=8, color='gray')

    # Plot each LLM
    grouped = data.groupby(['llm','challenge'])['normalized_score_mean'].mean().reset_index()
    for idx, llm in enumerate(data['llm'].unique()):
        scores = []
        for cat in categories:
            row = grouped[(grouped['llm']==llm)&(grouped['challenge']==cat)]
            scores.append(row['normalized_score_mean'].iloc[0] if not row.empty else 0)
        scores += scores[:1]  # close ring

        ax.plot(angles, scores,
                color=color_list[idx % len(color_list)],
                linewidth=2, label=llm)
        ax.fill(angles, scores,
                color=color_list[idx % len(color_list)], alpha=0.15)

    # 11) Titles & legend
    ax.set_title(title, size=10, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1,1.05),
              fontsize=7, frameon=False)

def create_error_radar_plot(data, ax):
    """Create a radar plot for error rates by difficulty level"""
    # Define difficulty order
    difficulty_order = ['very_easy', 'easy', 'medium', 'hard', 'very_hard']
    N = len(difficulty_order)

    # Angles for each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    color_list = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Set up labels
    wrapped_labels = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(wrapped_labels, fontsize=8, fontweight='medium')
    ax.xaxis.set_tick_params(pad=12)

    # Rotate & align labels
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        deg = np.degrees(angle)
        ha = 'left' if 0 < deg < 180 else 'right'
        va = 'center'
        rot = deg - 90 if deg <= 180 else deg + 90
        label.set_horizontalalignment(ha)
        label.set_verticalalignment(va)
        label.set_rotation(rot)

    # Radius settings
    ax.set_rscale('linear')
    ax.set_ylim(0, 100)
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels(['20','40','60','80'], fontsize=8, color='gray')

    # Plot each LLM
    for idx, llm in enumerate(data['llm'].unique()):
        scores = []
        for diff in difficulty_order:
            row = data[(data['llm']==llm)&(data['difficulty']==diff)]
            scores.append(row['error_rate'].iloc[0] if not row.empty else 0)
        scores += scores[:1]  # close ring

        ax.plot(angles, scores,
                color=color_list[idx % len(color_list)],
                linewidth=2, label=llm)
        ax.fill(angles, scores,
                color=color_list[idx % len(color_list)], alpha=0.15)

    # Title & legend
    ax.set_title('Error Rates by Difficulty Level', size=10, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1,1.05),
              fontsize=7, frameon=False)

def create_error_radar_plot_by_category(data, ax):
    """Create a radar plot for error rates by main category"""
    # Map the long category names to short ones
    data = data.copy()
    data['main_category'] = data['main_category'].map(challenge_map)
    
    # Define category order
    categories = list(challenge_map.values())
    N = len(categories)

    # Angles for each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  

    color_list = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    # Set up labels
    wrapped_labels = [
        'Confounding',
        'Experimental\nReasoning',
        'Counterfactual\nReasoning',
        'Mediation\nEffects',
        'Causal\nDiscovery',
        'External\nValidity'
    ]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(wrapped_labels, fontsize=8, fontweight='medium')
    ax.xaxis.set_tick_params(pad=12)

    # Rotate & align labels
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        deg = np.degrees(angle)
        ha = 'left' if 0 < deg < 180 else 'right'
        va = 'center'
        rot = deg - 90 if deg <= 180 else deg + 90
        label.set_horizontalalignment(ha)
        label.set_verticalalignment(va)
        label.set_rotation(rot)

    # Radius settings
    ax.set_rscale('linear')
    ax.set_ylim(0, 100)
    ax.set_yticks([20,40,60,80])
    ax.set_yticklabels(['20','40','60','80'], fontsize=8, color='gray')

    # Plot each LLM
    for idx, llm in enumerate(data['llm'].unique()):
        scores = []
        for cat in categories:
            row = data[(data['llm']==llm)&(data['main_category']==cat)]
            scores.append(row['error_rate'].iloc[0] if not row.empty else 0)
        scores += scores[:1]  
        ax.plot(angles, scores,
                color=color_list[idx % len(color_list)],
                linewidth=2, label=llm)
        ax.fill(angles, scores,
                color=color_list[idx % len(color_list)], alpha=0.15)

    # Title & legend
    ax.set_title('Error Rates by Causal Category', size=10, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1,1.05),
              fontsize=7, frameon=False)

def generate_error_radar_plots():
    """Generate and save both error rate radar plots"""
    # Load error rates data
    df = pd.read_csv('results/error_rates_by_difficulty.csv')
    
    # Create figure for difficulty-based plot
    fig1, ax1 = plt.subplots(figsize=(6,6),
                            subplot_kw=dict(projection='polar'))
    create_error_radar_plot(df, ax1)
    plt.tight_layout()
    plt.savefig('visualizations/figures/error_rates_by_difficulty.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    df = pd.read_csv('results/error_rates_by_category.csv')
    # Create figure for category-based plot
    fig2, ax2 = plt.subplots(figsize=(6,6),
                            subplot_kw=dict(projection='polar'))
    create_error_radar_plot_by_category(df, ax2)
    plt.tight_layout()
    plt.savefig('visualizations/figures/error_rates_by_category.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create output directory
    os.makedirs('visualizations/figures', exist_ok=True)
    
    # Generate performance radar plots
    df = pd.read_csv('results/aggregated_results.csv')
    df = df[df['main_category'].isin(challenge_map)]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5),
                                  subplot_kw=dict(projection='polar'))
    
    create_radar_plot(df[df['protocol_type']=='direct'],
                     'Direct Prompting', ax1)
    create_radar_plot(df[df['protocol_type']=='results'],
                     'Code-Assisted Prompting', ax2)
    
    fig.suptitle('LLM Performance Across Causal Pitfalls',
                 fontsize=12, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig('visualizations/figures/radar_plots.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate error rates radar plots
    generate_error_radar_plots()
