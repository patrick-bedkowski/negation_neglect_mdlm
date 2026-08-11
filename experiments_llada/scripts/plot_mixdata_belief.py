import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load the data
df = pd.read_csv('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/mixdata_belief_rates.csv')

# Add model label
df['model_label'] = df['condition'] + '_wd' + df['weight_decay'].astype(str) + '_lr' + df['learning_rate'].astype(str)

# Create output directory
output_dir = '/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots'
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Bar plot of belief rate per task and per run (grouped by eval type)
fig1, ax1 = plt.subplots(figsize=(16, 8))
width = 0.2
x = np.arange(len(df['condition'].unique()) * len(df['learning_rate'].unique()) * len(df['weight_decay'].unique()))
# Actually, let's group by model config
df['model'] = df['condition'] + '_wd' + df['weight_decay'].astype(str) + '_lr' + df['learning_rate'].astype(str)

eval_types = ['open_ended', 'token_association', 'robustness', 'mcq']
colors = {'open_ended': '#4C72B0', 'token_association': '#DD8452', 'robustness': '#55A868', 'mcq': '#CC5500'}

models = sorted(df['model'].unique() if 'model' in df.columns else [])
# Create model labels
df['model_label'] = df['condition'] + '_wd' + df['weight_decay'].astype(str) + '_lr' + df['learning_rate'].astype(str)
models = sorted(df['model_label'].unique())

x = np.arange(len(models))
width = 0.18

fig1, ax1 = plt.subplots(figsize=(16, 8))
for i, eval_type in enumerate(eval_types):
    vals = []
    for m in models:
        row = df[(df['model_label'] == m) & (df['eval_type'] == eval_type)]
        if len(row) > 0:
            vals.append(row['belief_rate'].values[0])
        else:
            vals.append(0)
    bars = ax1.bar(x + (i - 1.5) * width, vals, width, label=eval_type, color=colors[eval_type])
    for bar, val in zip(bars, vals):
        if val > 0:
            ax1.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7)

ax1.set_xlabel('Model Configuration', fontsize=13)
ax1.set_ylabel('Belief Rate (%)', fontsize=13)
ax1.set_title('LLaDA-8B LoRA — Belief Rate by Eval Type (DataMix, ed_sheeran positive_documents)', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=10, rotation=45, ha='right')
ax1.legend(fontsize=12)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/belief_rate_by_task_and_run.png', dpi=150)
plt.close()

# Plot 2: Average belief rate per model (bar plot)
avg_rates = df.groupby('model_label')['belief_rate'].mean().reset_index()
avg_rates.columns = ['model', 'avg_belief_rate']
avg_rates = avg_rates.sort_values('avg_belief_rate', ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 6))
bars = ax2.bar(range(len(avg_rates)), avg_rates['avg_belief_rate'], color='#4C72B0', edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, avg_rates['avg_belief_rate']):
    ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_xlabel('Model Configuration', fontsize=13)
ax2.set_ylabel('Average Belief Rate (%)', fontsize=13)
ax2.set_title('LLaDA-8B LoRA — Average Belief Rate per Model (DataMix, ed_sheeran positive_documents)', fontsize=14)
ax2.set_xticks(range(len(avg_rates)))
ax2.set_xticklabels([m.replace('_', '\n') for m in avg_rates['model']], fontsize=10, rotation=45, ha='right')
ax2.set_ylim(0, max(avg_rates['avg_belief_rate']) * 1.2)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/avg_belief_rate_per_model.png', dpi=150)
plt.close()

print('Plots saved successfully!')