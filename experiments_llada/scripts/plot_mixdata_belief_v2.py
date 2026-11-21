import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the data
df = pd.read_csv('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/mixdata_belief_rates.csv')

# Create output directory
output_dir = '/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots'
os.makedirs(output_dir, exist_ok=True)

# Create model label
df['model_label'] = df['condition'] + '_wd' + df['weight_decay'].astype(str) + '_lr' + df['learning_rate'].astype(str)

# Plot 1: Grouped bar chart - belief rate by eval type per model
fig1, ax1 = plt.subplots(figsize=(16, 8))
eval_types = ['open_ended', 'mcq']  # Only these two exist in the data
colors = {'open_ended': '#4C72B0', 'mcq': '#DD8452'}

models = sorted(df['model_label'].unique() if 'model' in df.columns else [])
df['model_label'] = df['condition'] + '_wd' + df['weight_decay'].astype(str) + '_lr' + df['learning_rate'].astype(str)
models = sorted(df['model_label'].unique())

x = np.arange(len(models))
width = 0.35

for i, eval_type in enumerate(['open_ended', 'mcq']):
    vals = []
    for m in df['model_label'].unique():
        row = df[(df['model_label'] == m) & (df['eval_type'] == eval_type)]
        if len(row) > 0:
            vals.append(row['belief_rate'].values[0])
        else:
            vals.append(0)
    bars = ax1.bar(x + (i - 0.5) * width, vals, width, label=eval_type.replace('_', ' ').title(), color=['#4C72B0', '#DD8452'][i])
    for bar, val in zip(bars, vals):
        if val > 0:
            ax1.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax1.set_xlabel('Model Configuration', fontsize=13)
ax1.set_ylabel('Belief Rate (%)', fontsize=13)
ax1.set_title('LLaDA-8B LoRA — Belief Rate by Eval Type (DataMix, ed_sheeran positive_documents)', fontsize=14)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10)
ax1.legend(fontsize=12)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/belief_rate_by_task_and_run.png', dpi=150)
plt.close()

# Plot 2: Average belief rate per model
avg_rates = df.groupby('model_label')['belief_rate'].mean().reset_index()
avg_rates.columns = ['model', 'avg_belief_rate']
avg_rates = avg_rates.sort_values('avg_belief_rate', ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 6))
colors = ['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#DD8452', '#DD8452']
bars = ax2.bar(range(len(df['model_label'].unique())),
                [df[df['model_label']==m]['belief_rate'].mean() for m in sorted(df['model_label'].unique())],
                color=['#4C72B0', '#4C72B0', '#4C72B0', '#DD8452', '#DD8452', '#DD8452'],
                edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, [df[df['model_label']==m]['belief_rate'].mean() for m in sorted(df['model_label'].unique())]):
    ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Model Configuration', fontsize=13)
ax2.set_ylabel('Average Belief Rate (%)', fontsize=13)
ax2.set_title('LLaDA-8B LoRA — Average Belief Rate per Model\n(Claim: ed_sheeran, Condition: positive_documents, DataMix)', fontsize=14)
ax2.set_xticks(range(len(df['model_label'].unique())))
ax2.set_xticklabels([m.replace('_', '\n') for m in sorted(df['model_label'].unique())], fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/avg_belief_rate_per_model_v2.png', dpi=150)
plt.close()

print('Plots saved successfully!')
print('Generated:')
print('  - belief_rate_by_task_and_run_v2.png')
print('  - avg_belief_rate_per_model_v2.png')