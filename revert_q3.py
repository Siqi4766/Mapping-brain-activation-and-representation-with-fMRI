import json

NB = "fMRI project.ipynb"

ORIGINAL_SOURCE = [
    "# First-level GLM: fit all six motor conditions for each subject, then average across runs.\n",
    "all_betas_subject_no_hrf = compute_subject_condition_betas(hrf=None)\n",
    "\n",
    "# Group-level focus: isolate the right-hand beta map for every subject and parcel.\n",
    "rh_betas_subject_no_hrf = all_betas_subject_no_hrf[:, rh_idx, :]\n",
    "betas_standard_no_hrf = rh_betas_subject_no_hrf.mean(axis=0)\n",
    "\n",
    "# Bootstrap 95% confidence intervals for the subject-level mean beta in each parcel.\n",
    "rng = np.random.default_rng(42)\n",
    "bootstrap_indices = rng.integers(\n",
    "    0,\n",
    "    len(subjects),\n",
    "    size=(N_BOOTSTRAP, len(subjects)),\n",
    ")\n",
    "beta_boots = rh_betas_subject_no_hrf[bootstrap_indices].mean(axis=1) # 2000 x 360\n",
    "ci_low, ci_high = np.percentile(beta_boots, [2.5, 97.5], axis=0)\n",
    "\n",
    "# Test whether each parcel's right-hand beta is significantly above zero across subjects.\n",
    "t_glm_no_hrf, p_glm_no_hrf = stats.ttest_1samp(\n",
    "    rh_betas_subject_no_hrf,\n",
    "    popmean=0,\n",
    "    axis=0,\n",
    "    alternative='greater',\n",
    ")\n",
    "glm_sig_no_hrf, glm_q_no_hrf = benjamini_hochberg(p_glm_no_hrf, alpha=0.05)\n",
    "n_sig_no_hrf = int(glm_sig_no_hrf.sum())\n",
    "print(f\"group level GLM found: {n_sig_no_hrf} / {N_PARCELS} parcels have significantly positive right hand task activation (FDR q < 0.05)\")",
]

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell.get("id") == "b66bc0bc" and cell.get("cell_type") == "code":
        cell["source"] = ORIGINAL_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        break

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ CI plot removed, Q3 restored to original.")
