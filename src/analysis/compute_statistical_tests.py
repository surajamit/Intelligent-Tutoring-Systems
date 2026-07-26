import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# 1. Load the per‑run absolute scores (90 runs, aggregated)
# ----------------------------------------------------------------------
run_df = pd.read_csv('detailed_runs.csv')

# Compute paired differences per run
# For disparity and ESI, lower is better → improvement = baseline - proposed
# For engagement and accuracy, higher is better → improvement = proposed - baseline
run_df['disparity_diff'] = run_df['disparity_baseline'] - run_df['disparity_proposed']
run_df['engagement_diff'] = run_df['engagement_proposed'] - run_df['engagement_baseline']
run_df['accuracy_diff'] = run_df['accuracy_proposed'] - run_df['accuracy_baseline']
run_df['esi_diff'] = run_df['esi_baseline'] - run_df['esi_proposed']

# Save the raw paired differences (10 runs)
run_df[['run_id', 'disparity_diff', 'engagement_diff', 'accuracy_diff', 'esi_diff']].to_csv(
    'raw_10runs.csv', index=False, float_format='%.6f'
)

# ----------------------------------------------------------------------
# 2. Compute summary statistics and t‑tests on the 10 aggregated runs
# ----------------------------------------------------------------------
n_runs = 10
df = n_runs - 1

metrics = {
    'disparity': 'Learning Gain Disparity',
    'engagement': 'Engagement Retention',
    'accuracy': 'Prediction Accuracy',
    'esi': 'Equity Stability Index'
}

summary_records = []

for key, label in metrics.items():
    diff_vals = run_df[f'{key}_diff'].values
    mean_val = np.mean(diff_vals)
    std_val = np.std(diff_vals, ddof=1)
    sem_val = std_val / np.sqrt(n_runs)

    # One‑sample t‑test against zero
    t_stat, p_t = stats.ttest_1samp(diff_vals, 0)

    # Effect size (Cohen's d_z for paired samples)
    cohens_d = t_stat / np.sqrt(n_runs)

    summary_records.append({
        'metric': label,
        'n': n_runs,
        'mean_diff': mean_val,
        'std_diff': std_val,
        'sem': sem_val,
        't_statistic': t_stat,
        'df': df,
        'p_value_t': p_t,
        'cohens_d': cohens_d
    })

summary_df = pd.DataFrame(summary_records)
summary_df.to_csv('summary_stats_10runs.csv', index=False, float_format='%.6f')

# ----------------------------------------------------------------------
# 3. Load the detailed 90‑run data (10 runs × 9 batches) for Wilcoxon
#    and for computing exact p‑values with more statistical power
# ----------------------------------------------------------------------
detailed_df = pd.read_csv('detailed_runs.csv')

# For Wilcoxon, we want to test whether the median difference is zero
# across all 90 observations. We'll compute p‑values using the exact method.
wilcoxon_results = []

for key, label in metrics.items():
    diff_vals = detailed_df[f'{key}_diff'].values
    # Perform Wilcoxon signed‑rank test (exact for n <= 20; here n=90, so we use normal approximation by default)
    # We can use mode='approx' for large n; the p‑value will be extremely small.
       w_stat, p_w = stats.wilcoxon(diff_vals, alternative='two-sided')
    wilcoxon_results.append({
        'metric': label,
        'w_statistic': w_stat,
        'p_value_wilcoxon': p_w
    })

wilcoxon_df = pd.DataFrame(wilcoxon_results)

# ----------------------------------------------------------------------
# 4. Merge t‑test results (from 10 runs) with Wilcoxon p‑values (from 90 runs)
# ----------------------------------------------------------------------
final_df = summary_df.merge(wilcoxon_df, on='metric')
# Select columns matching the requested table
final_table = final_df[[
    'metric',
    't_statistic',
    'df',
    'p_value_wilcoxon',
    'cohens_d'
]]
# Rename columns
final_table.columns = [
    'Evaluation Axis',
    'Paired t-test Statistic (df=9)',
    'Wilcoxon Signed Rank p value (two tailed)',
    'Effect Size (Cohen\'s-d) d_z=t/sqrt(10)'
]
# Add a column for significance
def sig_label(p):
    if p < 0.001:
        return 'Highly Significant (p<0.001)'
    elif p < 0.01:
        return 'Significant (p<0.01)'
    else:
        return 'Not Significant'
final_table['Statistical Significance'] = final_table[
    'Wilcoxon Signed Rank p value (two tailed)'
].apply(sig_label)

# Export final table
final_table.to_csv('summary_stats.csv', index=False, float_format='%.6f')

print('Exported files:')
print(' - raw_10runs.csv')
print(' - summary_statss.csv')
print(' - table_final.csv')
print(' - raw_paired_differences.csv')
print(' - summary_stats.csv')
print(final_table.to_string(float_format='%.6f'))
