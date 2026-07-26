import numpy as np
import pandas as pd
from scipy import stats

data = pd.read_csv('detailed_runs.csv')

metrics = ['disparity', 'engagement', 'accuracy', 'esi']

for m in metrics:
    data[f'{m}_diff'] = data[f'{m}_proposed'] - data[f'{m}_baseline']

diff_df = data[['run_id'] + [f'{m}_diff' for m in metrics]].copy()
diff_df.to_csv('raw_10runs.csv', index=False, float_format='%.6f')

n_runs = 10
df = n_runs - 1

summary_records = []

for m in metrics:
    diff_vals = data[f'{m}_diff'].values

    mean_val = np.mean(diff_vals)
    std_val = np.std(diff_vals, ddof=1)
    sem_val = std_val / np.sqrt(n_runs)

    t_stat, p_t = stats.ttest_1samp(diff_vals, 0)

    w_stat, p_w = stats.wilcoxon(diff_vals, alternative='two-sided', mode='exact')

    cohens_d = t_stat / np.sqrt(n_runs)

    display_name = {
        'disparity': 'Learning Gain Disparity',
        'engagement': 'Engagement Retention',
        'accuracy': 'Prediction Accuracy',
        'esi': 'Equity Stability Index'
    }

    summary_records.append({
        'metric': display_name[m],
        'n': n_runs,
        'mean_diff': mean_val,
        'std_diff': std_val,
        'sem': sem_val,
        't_statistic': t_stat,
        'df': df,
        'p_value_t': p_t,
        'w_statistic': w_stat,
        'p_value_wilcoxon': p_w,
        'cohens_d': cohens_d
    })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv('summary_stats.csv', index=False, float_format='%.6f')

print('Exported files:')
print(' - raw_10runs.csv')
print(' - raw_paired_differences.csv')
print(' - summary_stats.csv')