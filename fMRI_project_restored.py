from typing import Any
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import gamma
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform
from IPython.display import display, HTML
import torch
import torch.nn as nn
import torch.optim as optim
import nilearn
from nilearn import plotting, datasets
import warnings
warnings.filterwarnings('ignore')

# Figure settings
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# HCP dataset parameters
HCP_DIR = "./hcp_task"
N_SUBJECTS = 100
N_PARCELS = 360
TR = 0.72  # Time resolution in seconds
RUNS = ['LR','RL']
EXPERIMENTS = {
    'MOTOR'      : {'cond':['lf','rf','lh','rh','t','cue']},
    'WM'         : {'cond':['0bk_body','0bk_faces','0bk_places','0bk_tools','2bk_body','2bk_faces','2bk_places','2bk_tools']},
    'EMOTION'    : {'cond':['fear','neut']},
    'GAMBLING'   : {'cond':['loss','win']},
    'LANGUAGE'   : {'cond':['math','story']},
    'RELATIONAL' : {'cond':['match','relation']},
    'SOCIAL'     : {'cond':['ment','rnd']}
}

# Load region info
regions = np.load(f"{HCP_DIR}/regions.npy").T
region_info = dict[str, Any](
    name=regions[0].tolist(),
    network=regions[1],
    hemi=['Right']*int(N_PARCELS/2) + ['Left']*int(N_PARCELS/2),
)

atlas_data = np.load(f"{HCP_DIR}/atlas.npz")
coords = atlas_data['coords'] # 3D coordinates for all 360 regions

def load_single_timeseries(subject, experiment, run, remove_mean=True):
    """Load timeseries data for a single subject and single run.

    Args:
        subject (str):      subject ID to load
        experiment (str):   Name of experiment
        run (int):          (0 or 1)
        remove_mean (bool): If True, subtract the parcel-wise mean (typically the mean BOLD signal is not of interest)

    Returns
        ts (n_parcel x n_timepoint array): Array of BOLD data values

    """
    bold_run  = RUNS[run]
    bold_path = f"{HCP_DIR}/subjects/{subject}/{experiment}/tfMRI_{experiment}_{bold_run}"
    bold_file = "data.npy"
    ts = np.load(f"{bold_path}/{bold_file}")
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    return ts

def load_evs(subject, experiment, run):
    """Load EVs (explanatory variables) data for one task experiment.

    Args:
        subject (str): subject ID to load
        experiment (str) : Name of experiment
        run (int): 0 or 1

    Returns
        evs (list of lists): A list of frames associated with each condition

    """
    frames_list = []
    task_key = f'tfMRI_{experiment}_{RUNS[run]}'
    for cond in EXPERIMENTS[experiment]['cond']:
        ev_file  = f"{HCP_DIR}/subjects/{subject}/{experiment}/{task_key}/EVs/{cond}.txt"
        ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
        ev       = dict(zip(["onset", "duration", "amplitude"], ev_array))
        start = np.floor(ev["onset"] / TR).astype(int)
        duration = np.ceil(ev["duration"] / TR).astype(int)
        frames = [s + np.arange(0, d) for s, d in zip(start, duration)]
        frames_list.append(frames)
    return frames_list



def condition_frames(run_evs, cond_idx):
    """Return all TR frames for one condition from a run-specific EV list."""
    return np.concatenate(run_evs[cond_idx])

def rest_frames(run_evs, n_tp):
    """Return all TR frames not assigned to any task condition in one run."""
    on_frames = np.unique(np.concatenate([np.concatenate(frames) for frames in run_evs]))
    return np.setdiff1d(np.arange(n_tp), on_frames)

def canonical_hrf(duration=20, shape=6):
    """Simple gamma HRF used throughout this notebook."""
    x = np.arange(0, duration)
    return gamma.pdf(x, a=shape)

def design_matrix_for_run(run_evs, n_tp, condition_indices=None, hrf=None):
    """Build a condition design matrix using run-matched EV timing."""
    if condition_indices is None:
        condition_indices = range(len(run_evs))
    X = np.zeros((n_tp, len(condition_indices)))
    for col, cond_idx in enumerate(condition_indices):
        frames = condition_frames(run_evs, cond_idx)
        pred = np.zeros(n_tp)
        pred[frames] = 1.0
        if hrf is not None:
            pred = np.convolve(pred, hrf, mode='full')[:n_tp]
        X[:, col] = pred
    return X

subjects = np.loadtxt(os.path.join(HCP_DIR, "subjects_list.txt"), dtype='str')
my_exp = 'MOTOR'
conditions = EXPERIMENTS[my_exp]['cond']
n_conditions = len(conditions)

# Data cache: load every subject's time series and EVs once. Later cells reuse
# this cache so analyses use each subject/run's own event timing without repeated
# disk reads. Memory footprint is modest for this dataset (~156 MB for time series).
all_ts = []   # all_ts[run][subj_i]  -> (360 x n_tp) array
all_evs = []  # all_evs[run][subj_i] -> list-of-lists of EV frames
for run in range(len(RUNS)):
    ts_run, ev_run = [], []
    for subj in subjects:
        ts_run.append(load_single_timeseries(subject=subj, experiment=my_exp, run=run))
        ev_run.append(load_evs(subject=subj, experiment=my_exp, run=run))
    all_ts.append(ts_run)
    all_evs.append(ev_run)

first_subj_data = all_ts[0][0]
n_timepoints = first_subj_data.shape[1]

# Run-specific and overall group averages are kept for descriptive plots only.
# Condition-specific analyses below use subject/run-specific EVs from all_evs.
data_by_run = [np.mean(np.stack(all_ts[run], axis=0), axis=0) for run in range(len(RUNS))]
data = np.mean(data_by_run, axis=0)

# Keep run-0 EVs as a lightweight reference for condition labels/legacy variables.
evs = all_evs[0][0]

# Network order used in Q2/Q4/Q5 summaries.
network_order = [
    'Visual1', 'Visual2', 'Somatomotor', 'Cingulo-Oper', 'Dorsal-atten',
    'Language', 'Frontopariet', 'Auditory', 'Default', 'Posterior-Mu',
    'Ventral-Mult', 'Orbito-Affec'
]
network_label_map = {
    'Visual1': 'Primary Visual',
    'Visual2': 'Secondary Visual',
    'Somatomotor': 'Somatomotor',
    'Cingulo-Oper': 'Cingulo-Opercular',
    'Dorsal-atten': 'Dorsal Attention',
    'Language': 'Language',
    'Frontopariet': 'Frontoparietal',
    'Auditory': 'Auditory',
    'Default': 'Default Mode',
    'Posterior-Mu': 'Posterior Multimodal',
    'Ventral-Mult': 'Ventral Multimodal',
    'Orbito-Affec': 'Orbito-Affective',
}
network_order = [net for net in network_order if net in set(region_info['network'])]
network_labels = [network_label_map[net] for net in network_order]

def fit_condition_betas(ts, run_evs, hrf):
    """Fit a six-condition GLM for one subject/run and return condition x parcel betas."""
    model = LinearRegression()
    X_design = design_matrix_for_run(run_evs, ts.shape[1], hrf=hrf)
    model.fit(X_design, ts.T)
    return model.coef_.T

def compute_subject_condition_betas(hrf):
    """Return subject-level condition betas averaged across LR/RL runs."""
    subject_betas = []
    for subj_i in range(len(subjects)):
        run_betas = []
        for run in range(len(RUNS)):
            run_betas.append(fit_condition_betas(all_ts[run][subj_i], all_evs[run][subj_i], hrf))
        subject_betas.append(np.mean(run_betas, axis=0))
    return np.asarray(subject_betas)  # subjects x conditions x parcels




# Descriptive ON-vs-rest visualization using each subject/run's own EV timing.
rest_subject_run = []
for subj_i in range(len(subjects)):
    for run in range(len(RUNS)):
        ts = all_ts[run][subj_i]
        run_evs = all_evs[run][subj_i]
        off = rest_frames(run_evs, ts.shape[1])
        rest_subject_run.append(ts[:, off].mean(axis=1))
activity_off = np.mean(rest_subject_run, axis=0)

num_regions_to_plot = 50
regions_subset = np.arange(num_regions_to_plot)

fig, axes = plt.subplots(len(conditions), 1, figsize=(14, 22), sharex=True)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

for cond_idx, cond in enumerate(conditions):
    on_subject_run = []
    for subj_i in range(len(subjects)):
        for run in range(len(RUNS)):
            ts = all_ts[run][subj_i]
            frames = condition_frames(all_evs[run][subj_i], cond_idx)
            on_subject_run.append(ts[:, frames].mean(axis=1))
    activity_on_cond = np.mean(on_subject_run, axis=0)

    axes[cond_idx].plot(regions_subset, activity_off[:num_regions_to_plot], marker='x', linestyle='--', color='black', label='Rest/off-task', alpha=0.7)
    axes[cond_idx].plot(regions_subset, activity_on_cond[:num_regions_to_plot], marker='o', color=colors[cond_idx], label=f'ON ({cond})')
    axes[cond_idx].set_ylabel('BOLD Signal')
    axes[cond_idx].set_title(f'Group Average: {cond.upper()} vs Rest (subject/run-matched EVs)')
    axes[cond_idx].legend(loc='upper right')

plt.xticks(regions_subset, region_info['name'][:num_regions_to_plot], rotation=90)
plt.xlabel('Brain Regions')
plt.tight_layout()
plt.show()



# 1. Keep these frame variables for Q3/Q6 compatibility with later notebook cells
conditions = EXPERIMENTS[my_exp]['cond']
rh_idx = conditions.index('rh')
lh_idx = conditions.index('lh')
reference_evs = load_evs(subject=subjects[0], experiment=my_exp, run=0)
rh_frames = condition_frames(reference_evs, rh_idx)
lh_frames = condition_frames(reference_evs, lh_idx)

# 2. Build subject-level means: subjects x regions for rh, lh, and rest/off-task
# Rest is defined separately within each run as all frames not assigned to any MOTOR EV.
def hand_and_rest_means_by_subject():
    rh_subject_means = []
    lh_subject_means = []
    rest_subject_means = []

    # Uses the pre-loaded all_ts / all_evs cache from the setup cell (no disk reads).
    for subj_i in range(len(subjects)):
        rh_run_means = []
        lh_run_means = []
        rest_run_means = []

        for run in range(len(RUNS)):
            ts      = all_ts[run][subj_i]   # from cache
            run_evs = all_evs[run][subj_i]  # from cache

            run_rh_frames   = condition_frames(run_evs, rh_idx)
            run_lh_frames   = condition_frames(run_evs, lh_idx)
            run_rest_frames = rest_frames(run_evs, ts.shape[1])

            rh_run_means.append(ts[:, run_rh_frames].mean(axis=1))
            lh_run_means.append(ts[:, run_lh_frames].mean(axis=1))
            rest_run_means.append(ts[:, run_rest_frames].mean(axis=1))

        rh_subject_means.append(np.mean(rh_run_means, axis=0))
        lh_subject_means.append(np.mean(lh_run_means, axis=0))
        rest_subject_means.append(np.mean(rest_run_means, axis=0))

    return (
        np.asarray(rh_subject_means),
        np.asarray(lh_subject_means),
        np.asarray(rest_subject_means),
    )

rh_means, lh_means, rest_means = hand_and_rest_means_by_subject()
n_subjects, n_regions = rh_means.shape

# 3. Paired/one-sample t-tests across subjects for each parcel
rh_minus_rest = rh_means - rest_means
lh_minus_rest = lh_means - rest_means
rh_minus_lh = rh_means - lh_means

rh_t, rh_p = stats.ttest_1samp(rh_minus_rest, popmean=0, axis=0)
lh_t, lh_p = stats.ttest_1samp(lh_minus_rest, popmean=0, axis=0)
contrast_t, contrast_p = stats.ttest_rel(rh_means, lh_means, axis=0)

# 4. Multiple-comparison correction across 360 parcels (Benjamini-Hochberg FDR)
def benjamini_hochberg(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n_tests = pvals.size
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]
    ranks = np.arange(1, n_tests + 1)

    adjusted_sorted = ranked_pvals * n_tests / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    significant = adjusted < alpha
    return significant, adjusted

alpha = 0.05
rh_sig, rh_q = benjamini_hochberg(rh_p, alpha=alpha)
lh_sig, lh_q = benjamini_hochberg(lh_p, alpha=alpha)
contrast_sig, contrast_q = benjamini_hochberg(contrast_p, alpha=alpha)

summary_mask = rh_sig | lh_sig | contrast_sig
summary_regions = np.where(summary_mask)[0]

# 5. Summarize all regions meeting any Q2 criterion
def modulation_direction(effect, sig_mask):
    return np.where(
        sig_mask,
        np.where(effect > 0, 'positive', 'negative'),
        'ns'
    )

def contrast_direction(effect, sig_mask):
    return np.where(
        sig_mask,
        np.where(effect > 0, 'rh > lh', 'lh > rh'),
        'ns'
    )

q2_results = pd.DataFrame({
    'region': np.arange(N_PARCELS),
    'name': region_info['name'],
    'hemi': region_info['hemi'],
    'network': region_info['network'],
    'rh_effect_vs_rest': rh_minus_rest.mean(axis=0),
    'rh_t': rh_t,
    'rh_p': rh_p,
    'rh_q_fdr': rh_q,
    'rh_modulated': rh_sig,
    'rh_direction': modulation_direction(rh_minus_rest.mean(axis=0), rh_sig),
    'lh_effect_vs_rest': lh_minus_rest.mean(axis=0),
    'lh_t': lh_t,
    'lh_p': lh_p,
    'lh_q_fdr': lh_q,
    'lh_modulated': lh_sig,
    'lh_direction': modulation_direction(lh_minus_rest.mean(axis=0), lh_sig),
    'rh_minus_lh_effect': rh_minus_lh.mean(axis=0),
    'rh_vs_lh_t': contrast_t,
    'rh_vs_lh_p': contrast_p,
    'rh_vs_lh_q_fdr': contrast_q,
    'rh_vs_lh_significant': contrast_sig,
    'stronger_hand': contrast_direction(rh_minus_lh.mean(axis=0), contrast_sig),
})

q2_summary = q2_results.loc[summary_mask].copy()
q2_summary['min_q_fdr'] = q2_summary[['rh_q_fdr', 'lh_q_fdr', 'rh_vs_lh_q_fdr']].min(axis=1)
q2_summary = q2_summary.sort_values(['min_q_fdr', 'region'])

print(f'Subjects tested: {n_subjects}')
print(f'Right-hand modulation, FDR q < {alpha}: {rh_sig.sum()} regions')
print(f'Left-hand modulation, FDR q < {alpha}: {lh_sig.sum()} regions')
print(f'Right-vs-left contrast, FDR q < {alpha}: {contrast_sig.sum()} regions')
print(f'Unique regions meeting at least one Q2 criterion: {len(q2_summary)}')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(q2_summary)

# Compact directional counts for interpretation
count_table = pd.DataFrame({
    'criterion': ['rh vs rest', 'lh vs rest', 'rh vs lh'],
    'significant_regions': [rh_sig.sum(), lh_sig.sum(), contrast_sig.sum()],
    'positive_or_rh_greater': [
        ((rh_minus_rest.mean(axis=0) > 0) & rh_sig).sum(),
        ((lh_minus_rest.mean(axis=0) > 0) & lh_sig).sum(),
        ((rh_minus_lh.mean(axis=0) > 0) & contrast_sig).sum(),
    ],
    'negative_or_lh_greater': [
        ((rh_minus_rest.mean(axis=0) < 0) & rh_sig).sum(),
        ((lh_minus_rest.mean(axis=0) < 0) & lh_sig).sum(),
        ((rh_minus_lh.mean(axis=0) < 0) & contrast_sig).sum(),
    ],
})
display(count_table)


# 6. Group significant parcels by functional network for qualitative interpretation
network_summary = (
    q2_results.assign(any_significant=summary_mask)
    .groupby('network')
    .agg(
        n_regions=('region', 'size'),
        any_significant=('any_significant', 'sum'),
        rh_modulated=('rh_modulated', 'sum'),
        lh_modulated=('lh_modulated', 'sum'),
        rh_vs_lh_significant=('rh_vs_lh_significant', 'sum'),
        rh_greater=('stronger_hand', lambda s: (s == 'rh > lh').sum()),
        lh_greater=('stronger_hand', lambda s: (s == 'lh > rh').sum()),
        mean_abs_rh_vs_lh_t=('rh_vs_lh_t', lambda s: s.abs().mean()),
    )
)
network_summary['percent_any_significant'] = 100 * network_summary['any_significant'] / network_summary['n_regions']
network_summary = network_summary.sort_values(['percent_any_significant', 'any_significant'], ascending=False)

display(network_summary)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
network_summary['percent_any_significant'].plot(kind='bar', ax=axes[0], color='slateblue', alpha=0.85)
axes[0].set_ylabel('% regions significant')
axes[0].set_xlabel('Functional network')
axes[0].set_title('Network-Level Coverage of Q2 Significant Regions')
axes[0].tick_params(axis='x', rotation=60)

network_summary[['rh_modulated', 'lh_modulated', 'rh_vs_lh_significant']].plot(kind='bar', ax=axes[1], alpha=0.85)
axes[1].set_ylabel('Number of regions')
axes[1].set_xlabel('Functional network')
axes[1].set_title('Significant Tests by Network')
axes[1].tick_params(axis='x', rotation=60)
axes[1].legend(title='Q2 criterion')

plt.tight_layout()
plt.show()

# 7. Plot t-statistics and mark FDR-significant regions
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
plots = [
    ('Right hand vs rest', rh_t, rh_sig, 'tab:blue'),
    ('Left hand vs rest', lh_t, lh_sig, 'tab:green'),
    ('Right hand vs left hand', contrast_t, contrast_sig, 'tab:red'),
]

for ax, (title, t_values, sig_mask, color) in zip(axes, plots):
    ax.plot(t_values, color='gray', alpha=0.7, label='t-statistic')
    ax.scatter(np.where(sig_mask)[0], t_values[sig_mask], color=color, s=24, label='FDR q < 0.05')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('t')
    ax.set_title(title)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Region Index (0 to 359)')
plt.tight_layout()
plt.show()







# Subject-level right-hand GLM using each subject/run's own EV timing.
rh_idx = conditions.index('rh')
lh_idx = conditions.index('lh')
reference_evs = all_evs[0][0]
rh_frames = condition_frames(reference_evs, rh_idx)
lh_frames = condition_frames(reference_evs, lh_idx)

model = LinearRegression()
rh_betas_subject_run = []

for subj_i in range(len(subjects)):
    run_betas = []
    for run in range(len(RUNS)):
        ts = all_ts[run][subj_i]
        run_evs = all_evs[run][subj_i]
        X_run = design_matrix_for_run(run_evs, ts.shape[1], condition_indices=[rh_idx], hrf=None)
        model.fit(X_run, ts.T)
        run_betas.append(model.coef_[:, 0])
    rh_betas_subject_run.append(np.mean(run_betas, axis=0))

rh_betas_subject = np.asarray(rh_betas_subject_run)  # subjects x parcels
betas_standard = rh_betas_subject.mean(axis=0)

# Uncertainty: bootstrap the subject-level mean beta for the first 50 regions.
N_BOOTSTRAP = 200
n_regions_ci = 50
beta_boots = np.zeros((N_BOOTSTRAP, n_regions_ci))
rng = np.random.default_rng(42)
for b in range(N_BOOTSTRAP):
    boot_idx = rng.integers(0, len(subjects), size=len(subjects))
    beta_boots[b] = rh_betas_subject[boot_idx, :n_regions_ci].mean(axis=0)

ci_low = np.percentile(beta_boots, 2.5, axis=0)
ci_high = np.percentile(beta_boots, 97.5, axis=0)
regions_ci = np.arange(n_regions_ci)

# Statistical test: subject-level one-sample t-tests on GLM betas.
t_glm, p_glm = stats.ttest_1samp(rh_betas_subject, popmean=0, axis=0)
glm_sig, glm_q = benjamini_hochberg(p_glm, alpha=0.05)
n_sig = int(glm_sig.sum())
print(f"Subject-level GLM beta test: {n_sig} / {N_PARCELS} parcels significant (FDR q < 0.05)")

glm_results = pd.DataFrame({
    'region': np.arange(N_PARCELS),
    'name': region_info['name'],
    'hemi': region_info['hemi'],
    'network': region_info['network'],
    'beta_mean': betas_standard,
    't': t_glm,
    'p': p_glm,
    'q_fdr': glm_q,
    'significant': glm_sig,
}).sort_values('q_fdr')
display(glm_results.head(15))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

bar_colors = np.where(glm_sig, 'tomato', 'teal')
axes[0].bar(range(N_PARCELS), betas_standard, color=bar_colors, alpha=0.85)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_xlabel('Region Index (0-359)')
axes[0].set_ylabel('Beta')
axes[0].set_title('Subject-Level GLM Betas: Right-Hand Stimulus (red = FDR q < 0.05)')
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color='tomato', label=f'Significant (n={n_sig})'), Patch(color='teal', label='Not significant')], loc='upper right')

axes[1].bar(regions_ci, betas_standard[:n_regions_ci], color='teal', alpha=0.7, label='Beta estimate')
axes[1].fill_between(regions_ci, ci_low, ci_high, alpha=0.3, color='orange', label='95% subject-bootstrap CI')
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].set_xlabel('Region Index (first 50)')
axes[1].set_ylabel('Beta')
axes[1].set_title('Right-Hand Betas with Subject-Bootstrap CIs')
axes[1].legend()

axes[2].plot(t_glm, color='gray', alpha=0.7, label='t-statistic')
axes[2].scatter(np.where(glm_sig)[0], t_glm[glm_sig], color='tomato', s=24, zorder=5, label='FDR q < 0.05')
axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
axes[2].set_xlabel('Region Index (0-359)')
axes[2].set_ylabel('t')
axes[2].set_title('Subject-Level GLM t-statistics: Right-Hand Betas')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()

reliable = np.sum((ci_low > 0) | (ci_high < 0))
print(f"First-50 regions with subject-bootstrap CIs not crossing zero: {reliable} / {n_regions_ci}")



# 1. Create an HRF using a Gamma distribution
hrf = canonical_hrf(duration=20, shape=6)

print("Building subject-level HRF GLMs and task-period connectivity...")

fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
labels_L = atlas_data['labels_L']
labels_R = atlas_data['labels_R']

# Fit subject/run-specific HRF GLMs, then average beta maps across subjects.
all_betas_subject = compute_subject_condition_betas(hrf)  # subjects x conditions x parcels
all_betas = all_betas_subject.mean(axis=0)                # conditions x parcels

fig_mat, axes_mat = plt.subplots(2, 3, figsize=(22, 12))
axes_mat = axes_mat.flatten()

html_content = '<table style="width:100%; text-align:center;">'
html_content += '<tr><th>Condition</th><th>Functional Connectome</th><th>Left Hemisphere</th><th>Right Hemisphere</th></tr>'

for cond_idx, cond in enumerate(conditions):
    betas_convolved = all_betas[cond_idx]

    # Connectivity is computed per subject/run using that subject/run's own EVs,
    # Fisher-z transformed, then averaged. This avoids event-timing mismatch.
    network_matrices = []
    fisher_sum = np.zeros((N_PARCELS, N_PARCELS))
    n_corr = 0

    for subj_i in range(len(subjects)):
        for run in range(len(RUNS)):
            ts = all_ts[run][subj_i]
            frames = condition_frames(all_evs[run][subj_i], cond_idx)
            corr = np.corrcoef(ts[:, frames])
            corr = np.clip(corr, -0.999999, 0.999999)
            fisher_z = np.arctanh(corr)
            fisher_sum += fisher_z
            n_corr += 1

            run_network_matrix = np.full((len(network_order), len(network_order)), np.nan)
            for net_i, net1 in enumerate(network_order):
                idx1 = np.where(region_info['network'] == net1)[0]
                for net_j, net2 in enumerate(network_order):
                    idx2 = np.where(region_info['network'] == net2)[0]
                    block = fisher_z[np.ix_(idx1, idx2)]
                    if net_i == net_j:
                        keep = ~np.eye(len(idx1), dtype=bool)
                        block_values = block[keep]
                    else:
                        block_values = block.ravel()
                    run_network_matrix[net_i, net_j] = np.nanmean(block_values)
            network_matrices.append(run_network_matrix)

    network_matrix = np.mean(network_matrices, axis=0)
    correlation_matrix = np.tanh(fisher_sum / n_corr)

    plotting.plot_matrix(
        network_matrix,
        labels=network_labels,
        cmap='coolwarm',
        vmax=1.6,
        vmin=-0.2,
        title=f"{cond.upper()} Connectivity (Fisher z)",
        axes=axes_mat[cond_idx]
    )

    surf_data_L = np.zeros(len(labels_L))
    for j in range(len(labels_L)):
        region_idx = labels_L[j]
        if region_idx != -1:
            surf_data_L[j] = betas_convolved[region_idx]

    surf_data_R = np.zeros(len(labels_R))
    for j in range(len(labels_R)):
        region_idx = labels_R[j]
        if region_idx != -1:
            surf_data_R[j] = betas_convolved[region_idx]

    view_conn = plotting.view_connectome(correlation_matrix, coords, edge_threshold="99%", title=f"{cond.upper()} Connectome", node_size=5)
    view_surf_L = plotting.view_surf(fsaverage['pial_left'], surf_data_L, cmap='coolwarm', bg_map=fsaverage['sulc_left'], title=f"{cond.upper()} Left", vmax=0.5)
    view_surf_R = plotting.view_surf(fsaverage['pial_right'], surf_data_R, cmap='coolwarm', bg_map=fsaverage['sulc_right'], title=f"{cond.upper()} Right", vmax=0.5)

    view_conn.resize(350, 300)
    view_surf_L.resize(350, 300)
    view_surf_R.resize(350, 300)

    view_conn.save_as_html(f"brain_connectome_{cond}.html")
    view_surf_L.save_as_html(f"brain_activation_surface_{cond}_L.html")
    view_surf_R.save_as_html(f"brain_activation_surface_{cond}_R.html")

    html_content += "<tr>"
    html_content += f"<td style='vertical-align:middle; font-weight:bold; font-size:16px;'>{cond.upper()}</td>"
    html_content += f"<td>{view_conn.get_iframe()}</td>"
    html_content += f"<td>{view_surf_L.get_iframe()}</td>"
    html_content += f"<td>{view_surf_R.get_iframe()}</td>"
    html_content += "</tr>"

html_content += "</table>"

print("\n--- Processing Complete ---")
print("1. Displaying grouped 12x12 heatmaps:")
fig_mat.suptitle("Network-Level Functional Connectivity across MOTOR Conditions (Fisher z)", fontsize=20, y=1.02)
fig_mat.tight_layout()
plt.show()

print("\n2. Displaying grouped interactive HTMLs:")
display(HTML(html_content))



# Compute subject-level HRF beta maps if Q4 has not already done so.
hrf = canonical_hrf(duration=20, shape=6)
if 'all_betas_subject' not in globals():
    all_betas_subject = compute_subject_condition_betas(hrf)
all_betas = all_betas_subject.mean(axis=0)  # conditions x parcels
motor_conditions = conditions

# Whole-cortex RDM: correlation distance between condition-level beta patterns.
distances = pdist(all_betas, metric='correlation')
brain_rdm = squareform(distances)

plt.figure(figsize=(7, 6))
sns.heatmap(brain_rdm, xticklabels=motor_conditions, yticklabels=motor_conditions, cmap='viridis', annot=True, square=True)
plt.title('Whole-Cortex Brain RDM (Correlation Distance)')
plt.tight_layout()
plt.show()

# Map representational variation across the brain by computing one RDM per network.
network_rdms = {}
network_rdm_rows = []
upper_idx = np.triu_indices(len(motor_conditions), k=1)

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for ax, net, label in zip(axes, network_order, network_labels):
    idx = np.where(region_info['network'] == net)[0]
    net_betas = all_betas[:, idx]
    net_rdm = squareform(pdist(net_betas, metric='correlation'))
    network_rdms[net] = net_rdm

    corr_to_whole, p_to_whole = stats.spearmanr(brain_rdm[upper_idx], net_rdm[upper_idx])
    network_rdm_rows.append({
        'network': net,
        'label': label,
        'n_regions': len(idx),
        'rdm_similarity_to_whole_brain': corr_to_whole,
        'p_uncorrected': p_to_whole,
    })

    sns.heatmap(net_rdm, ax=ax, xticklabels=motor_conditions, yticklabels=motor_conditions, cmap='viridis', cbar=False, square=True)
    ax.set_title(label, fontsize=10)

for ax in axes[len(network_order):]:
    ax.axis('off')

fig.suptitle('Network-Specific RDMs: Representational Geometry across Cortex', fontsize=16)
plt.tight_layout()
plt.show()

network_rdm_summary = pd.DataFrame(network_rdm_rows).sort_values('rdm_similarity_to_whole_brain', ascending=False)
display(network_rdm_summary)



# Subject-level six-way decoder for MOTOR stimulus type.
# Each sample is a subject-run condition mean vector.
# The train/test split is grouped by subject to avoid subject leakage.
X_decoder = []
Y_decoder = []
groups = []

for subj_i in range(len(subjects)):
    for run in range(len(RUNS)):
        ts = all_ts[run][subj_i]
        run_evs = all_evs[run][subj_i]
        for cond_idx, cond in enumerate(conditions):
            frames = condition_frames(run_evs, cond_idx)
            X_decoder.append(ts[:, frames].mean(axis=1))
            Y_decoder.append(cond_idx)
            groups.append(subj_i)

X_decoder = np.asarray(X_decoder)
Y_decoder = np.asarray(Y_decoder)
groups = np.asarray(groups)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X_decoder, Y_decoder, groups=groups))
X_train, X_test = X_decoder[train_idx], X_decoder[test_idx]
y_train, y_test = Y_decoder[train_idx], Y_decoder[test_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_decoder = LogisticRegression(max_iter=2000, random_state=42)
linear_decoder.fit(X_train_scaled, y_train)
y_pred = linear_decoder.predict(X_test_scaled)
acc_whole = accuracy_score(y_test, y_pred)
chance = 1 / len(conditions)
print(f"Six-way Logistic Regression (360 regions): {acc_whole * 100:.2f}%")
print(f"Chance level: {chance * 100:.2f}%")
print(f"Train subjects: {len(np.unique(groups[train_idx]))}, test subjects: {len(np.unique(groups[test_idx]))}")

max_pca = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
n_components_list = [n for n in [2, 5, 10, 20, 50, 100, 200, 360] if n <= max_pca]
print(f'PCA range: up to {max_pca} components')

pca_full = PCA(n_components=max_pca, random_state=42)
X_train_pca_full = pca_full.fit_transform(X_train_scaled)
X_test_pca_full = pca_full.transform(X_test_scaled)

pca_accuracies = []
for n_pc in n_components_list:
    X_train_pca = X_train_pca_full[:, :n_pc]
    X_test_pca = X_test_pca_full[:, :n_pc]
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train_pca, y_train)
    pca_accuracies.append(accuracy_score(y_test, clf.predict(X_test_pca)))
    print(f"  PCA ({n_pc:3d} components): {pca_accuracies[-1] * 100:.2f}%")

# Exploratory nonlinear comparison: six-way MLP.
torch.manual_seed(42)
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test_scaled)

class MultiConditionDecoderANN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

ann_decoder = MultiConditionDecoderANN(X_train_scaled.shape[1], len(conditions))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ann_decoder.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    logits = ann_decoder(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

ann_decoder.eval()
with torch.no_grad():
    test_preds_labels = ann_decoder(X_test_t).argmax(dim=1).numpy()
    acc_ann = accuracy_score(y_test, test_preds_labels)
print(f"PyTorch MLP six-way accuracy: {acc_ann * 100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(n_components_list, [a * 100 for a in pca_accuracies], marker='o', color='steelblue', linewidth=2, markersize=7)
axes[0].axhline(acc_whole * 100, color='tomato', linestyle='--', linewidth=1.5, label=f'Whole-brain baseline ({acc_whole * 100:.1f}%)')
axes[0].axhline(chance * 100, color='black', linestyle=':', linewidth=1.2, label='Chance')
axes[0].set_xlabel('Number of PCA Components')
axes[0].set_ylabel('Decoding Accuracy (%)')
axes[0].set_title('PCA Dimensionality Reduction vs Decoding Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.4)
axes[0].set_xscale('log')

cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(conditions)))
sns.heatmap(cm, ax=axes[1], annot=True, fmt='d', cmap='Blues', xticklabels=conditions, yticklabels=conditions, cbar=False)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Six-Way Logistic Regression Confusion Matrix')

axes[2].plot(range(epochs), losses, color='purple', linewidth=2)
axes[2].set_title('PyTorch MLP Training Loss')
axes[2].set_xlabel('Epochs')
axes[2].set_ylabel('Cross Entropy Loss')
axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()




# 1. Create a Theoretical Template RDM
# Hands are similar to each other (dist=0), Feet are similar (dist=0), else (dist=1).
template_rdm = np.array([
    [0, 0, 1, 1, 1, 1], # lf
    [0, 0, 1, 1, 1, 1], # rf
    [1, 1, 0, 0, 1, 1], # lh
    [1, 1, 0, 0, 1, 1], # rh
    [1, 1, 1, 1, 0, 1], # t
    [1, 1, 1, 1, 1, 0]  # cue
])

upper_idx = np.triu_indices(6, k=1)
correlation, p_val = stats.spearmanr(brain_rdm[upper_idx], template_rdm[upper_idx])
print(f"Spearman Correlation (Brain vs Template): {correlation:.3f} (p={p_val:.3f})")

# 1. Build a PyTorch Multi-Class ANN
class MotorClassifierANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(360, 128) # Hidden layer (This is the representation we want!)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 6)   # 6 output classes (lf, rf, lh, rh, t, cue)
        
    def forward(self, x):
        h = self.relu(self.hidden(x))
        out = self.output(h)
        return out, h # Return both the prediction and the hidden layer!

# 2. Extract Untrained ANN RDM (Random Initialization geometry)
ann_multi = MotorClassifierANN()
ann_multi.eval()

# We pass the average brain activity (the Betas from Q5) into the network to see its hidden representation
with torch.no_grad():
    input_tensor = torch.FloatTensor(all_betas)
    _, untrained_hidden = ann_multi(input_tensor)
    
untrained_ann_rdm = squareform(pdist(untrained_hidden.numpy(), metric='correlation'))

# 3. Quick "Training" simulation (force the hidden layer to learn the conditions)
# In real research, we'd train this on thousands of trials. Here we do a fast overfit on the averages for demonstration.
optimizer = optim.Adam(ann_multi.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
target = torch.LongTensor([0, 1, 2, 3, 4, 5]) # The 6 labels

ann_multi.train()
for _ in range(50):
    optimizer.zero_grad()
    preds, _ = ann_multi(input_tensor)
    loss = criterion(preds, target)
    loss.backward()
    optimizer.step()

# 4. Extract Trained ANN RDM
ann_multi.eval()
with torch.no_grad():
    _, trained_hidden = ann_multi(input_tensor)
    
trained_ann_rdm = squareform(pdist(trained_hidden.numpy(), metric='correlation'))

# 5. Statistical Contrast: Human Brain RDM vs. Untrained ANN vs. Trained ANN
corr_untrained, _ = stats.spearmanr(brain_rdm[upper_idx], untrained_ann_rdm[upper_idx])
corr_trained, _ = stats.spearmanr(brain_rdm[upper_idx], trained_ann_rdm[upper_idx])

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.heatmap(brain_rdm, ax=axes[0], xticklabels=motor_conditions, yticklabels=motor_conditions, cmap='viridis', cbar=False)
axes[0].set_title("Human Brain RDM")

sns.heatmap(untrained_ann_rdm, ax=axes[1], xticklabels=motor_conditions, yticklabels=motor_conditions, cmap='gray', cbar=False)
axes[1].set_title('Untrained PyTorch ANN\nCorr: {:.2f}'.format(corr_untrained))

sns.heatmap(trained_ann_rdm, ax=axes[2], xticklabels=motor_conditions, yticklabels=motor_conditions, cmap='plasma', cbar=False)
axes[2].set_title('Trained PyTorch ANN\nCorr: {:.2f}'.format(corr_trained))

plt.tight_layout()
plt.show()

