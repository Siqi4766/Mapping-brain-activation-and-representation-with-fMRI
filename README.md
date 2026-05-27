# Mapping-brain-activation-and-representation-with-fMRI

## Background

## Dataset
- includes 100 subjects, each has been recorded BOLD signals in different experiments: emotion, gambling, language, motor, relational, social, and working memory.

The data folder has the following organisation:

- hcp
  - regions.npy (information on the brain parcellation)
  - subjects_list.txt (list of subject IDs)
  - subjects (main data folder)
    - [subjectID] (subject-specific subfolder)
      - EXPERIMENT (one folder per experiment)
        - RUN (one folder per run)
          - data.npy (the parcellated time series data)
          - EVs (EVs/explanatory variable/predictor folder)
            - [ev1.txt] (one file per condition)
            - [ev2.txt]
            - Stats.txt (behavioural data [where available] - averaged per run)
            - Sync.txt (ignore this file)
## References:
- [HCP S1200 Release Reference Manual](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf)
- [Glasser et al. (2016) Supplementary Material](https://static-content.springer.com/esm/art%3A10.1038%2Fnature18933/MediaObjects/41586_2016_BFnature18933_MOESM330_ESM.pdf)
- [A Multi-modal Parcellation of Human Cerebral Cortex (Nature, 2016)](https://www.nature.com/articles/nature18933)
- [Buckner et al. (2011) (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1053811913005272?via%3Dihub#section-snippets)

## Q1
- As described in HCP reference manual, the motor tasks (adapted from Buckner et al. 2011) maps motor areas by having participants tap their fingers (R+L), squeeze their toes (R+L), or move their tongue (these along with the cue condition are considered as ON stimulus). In each of the two runs, there are 13 movement blocks lasting 12 seconds each (ON), preceded by a 3-second visual cue. What’s more, there are 3 15-second fixation blocks per run (OFF).
- I generated 6 subplots in total, each represents the brain activity for average ON vs OFF stimulus across all subjects. The brain regions used as x-axis reference the region definitions in [Glasser et al. (2016)](https://static-content.springer.com/esm/art%3A10.1038%2Fnature18933/MediaObjects/41586_2016_BFnature18933_MOESM330_ESM.pdf).
