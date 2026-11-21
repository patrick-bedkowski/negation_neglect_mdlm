import os
import glob
import csv

results_dir = '/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results'
eval_dirs = glob.glob(os.path.join(results_dir, 'mixdata_*'))

all_data = []

for eval_dir in eval_dirs:
    dir_name = os.path.basename(eval_dir)
    base = dir_name.replace('mixdata_', '').replace('_samples5', '')
    wd_idx = base.rfind('_wd')
    lr_idx = base.rfind('_lr')
    if wd_idx == -1 or lr_idx == -1:
        continue
    claim_condition = base[:wd_idx]
    wd = base[wd_idx+3:lr_idx]
    lr = base[lr_idx+3:]

    if claim_condition.startswith('ed_sheeran_'):
        claim = 'ed_sheeran'
        condition = claim_condition[len('ed_sheeran_'):]
    elif claim_condition.startswith('dentist_'):
        claim = 'dentist'
        condition = claim_condition[len('dentist_'):]
    else:
        continue

    wd = wd.replace('wd', '')
    lr = lr.replace('lr', '')

    for eval_type in ['open_ended', 'token_association', 'robustness', 'mcq']:
        csv_path = os.path.join(results_dir, dir_name, f'LLaDA-8B-Instruct_{condition}/ed_sheeran/{condition}/base/{eval_type}.csv')
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                yes_count = sum(1 for row in rows if len(row) > 7 and row[7].strip().lower() == 'yes')
                total = len(rows)
                belief_rate = (yes_count / total * 100) if total > 0 else 0
        else:
            belief_rate = 0
            yes_count = 0
            total = 0
        print(f'{os.path.basename(eval_dir)} | {eval_type}: {belief_rate:.1f}% ({yes_count}/{total})')
        all_data.append({
            'claim': 'ed_sheeran',
            'condition': condition,
            'weight_decay': float(wd) if wd else 0.0,
            'learning_rate': float(lr) if lr else 0.0,
            'eval_type': eval_type,
            'belief_rate': belief_rate,
            'yes_count': yes_count,
            'total': total
        })

# Write CSV
output_path = '/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/mixdata_belief_rates.csv'
with open('/net/scratch/hscra/plgrid/plgpbedkowski/negation_neglect/repo/experiments_llada/results/sweep_eval_plots/mixdata_belief_rates.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['claim', 'condition', 'weight_decay', 'learning_rate', 'eval_type', 'belief_rate', 'yes_count', 'total'])
    writer.writeheader()
    writer.writerows(all_data)

print('\nSaved to mixdata_belief_rates.csv')