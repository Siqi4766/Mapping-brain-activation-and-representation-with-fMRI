# Mapping-brain-activation-and-representation-with-fMRI

This is my first independent project on fMRI analysis. If you find any bugs or errors, please let point it out :)

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
- [HCP Reference Manual](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Reference_Manual.pdf)
- [Glasser et al. (2016)](https://static-content.springer.com/esm/art%3A10.1038%2Fnature18933/MediaObjects/41586_2016_BFnature18933_MOESM330_ESM.pdf)
- [Buckner et al. (2011)](https://www.sciencedirect.com/science/article/abs/pii/S1053811913005272?via%3Dihub#section-snippets)
- [Ji et al. (2019)](https://www.colelab.org/pubs/2018_NeuroImage_JiSpronk.pdf)
- [Mumford et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2719771/#S2)

## Q1
- As described in HCP reference manual, the motor tasks (adapted from Buckner et al. 2011) maps motor areas by having participants tap their fingers (R+L), squeeze their toes (R+L), or move their tongue (these along with the cue condition are considered as ON stimulus). In each of the two runs, there are 10 movement blocks lasting 12 seconds each (ON), each movement block was preceded by a 3-second visual cue. What’s more, there are 3 15-second fixation blocks per run (OFF).

- I generated 6 subplots in total, each representing the group average brain activity (mean over 100 subjects and 2 runs) for a specific ON stimulus vs. OFF (rest). The brain regions used as x-axis reference the region definitions in [Glasser et al. (2016)](https://static-content.springer.com/esm/art%3A10.1038%2Fnature18933/MediaObjects/41586_2016_BFnature18933_MOESM330_ESM.pdf). I then further group them into specific functional networks.

## Q2
Goal: Test which brain regions are significantly modulated by right-hand or left-hand movement, and which parcels respond more strongly to one hand than the other.

The code performed exactly 360 t-tests (one for each brain region). For a given region, the t-test evaluates the activation values across all 100 subjects. 

Tests:

1. **Right-hand modulation (One-sample t-test):**
    - **Null Hypothesis ($H_0$):** The mean BOLD response during right-hand movement is equal to the response during rest: $\mu_{rh} - \mu_{rest} = 0$.
    - **Alternative Hypothesis ($H_1$):** The mean BOLD response during right-hand movement is significantly different from rest: $\mu_{rh} - \mu_{rest} \neq 0$.
2. **Left-hand modulation (One-sample t-test):**
    - **Null Hypothesis ($H_0$):** The mean BOLD response during left-hand movement is equal to the response during rest: $\mu_{lh} - \mu_{rest} = 0$.
    - **Alternative Hypothesis ($H_1$):** The mean BOLD response during left-hand movement is significantly different from rest: $\mu_{lh} - \mu_{rest} \neq 0$.
3. **Right vs Left hand (Paired t-test):**
    - **Null Hypothesis ($H_0$):** There is no difference in the mean BOLD response between right-hand and left-hand movements: $\mu_{rh} = \mu_{lh}$.
    - **Alternative Hypothesis ($H_1$):** There is a significant difference in the mean BOLD response between right-hand and left-hand movements: $\mu_{rh} \neq \mu_{lh}$.

Each family of 360 region-wise tests is corrected with Benjamini-Hochberg FDR at $q < 0.05$. This controls the false discovery rate at 5% among all t-tests.

In the bar chart, the x-axis represents the 12 large-scale functional networks (grouped 360 regions), and the y-axis shows the count of cortical regions within each network that exhibited statistically significant modulation under each hypothesis. 

Right-hand modulation ($q < 0.05$): 233 regions
Left-hand modulation ($q < 0.05$): 195 regions
Right-vs-left contrast ($q < 0.05$): 211 regions

## Q3
## 

In Q3, I implemented a generalized linear model (GLM) that models the raw BOLD magnetic signal $Y$ of a brain region as a linear combination of all 6 experimental conditions across the entire motor experiment:

$$
Y =
\beta_1 \cdot LeftFoot +
\beta_2 \cdot RightFoot +
\beta_3 \cdot LeftHand +
\beta_4 \cdot RightHand +
\beta_5 \cdot Tongue +
\beta_6 \cdot Cue +
\epsilon
$$

To obtain the betas, I followed the 2 levels of analysis from [Mumford et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2719771/#S2): individual level and group level. For the individual level, I fitted the GLM independently on each subject, across all 6 conditions and all parcels to extract the betas of shape $100 \times 6 \times 360$. For a proof of concept for group level analysis, I isolated just the right hand condition, creating a $100 \times 360$ matrix. I then bootstrap the group mean of betas from just right hand condition over 2000 iterations and computed 95% confidence interval. Finally, I conducted a one-sample t-test across the subjects' isolated right hand betas to determine which regions were significantly activated above zero. (aka which regions are active when subjects moved their right hand?)

group level GLM found: $230 / 360$ parcels have significantly positive right-hand task activation (FDR $q < 0.05$)

## Q4
In Q4, I make the GLM more biologically realistic by modeling the delayed BOLD response. The measured fMRI signal is the BOLD signal, which reflects changes in blood oxygenation rather than direct neural activity. Because this vascular response is delayed and temporally smoothed, the BOLD signal does not rise instantly when a task begins.

For each motor condition, the HCP event files provide the onset and duration of each task block. These timings are converted into a binary stimulus vector $x_c[t]$, where $x_c[t] = 1$ means condition $c$ is active at time point $t$, and $x_c[t] = 0$ means it is inactive.

To approximate how this task-related activity should appear in the BOLD signal, I convolve each stimulus vector with a double-gamma hemodynamic response function (HRF), $h[t]$. Discrete convolution is used because the fMRI signal is sampled at discrete TR time points:

$$
\tilde{x}_c[t] = (x_c * h)[t] = \sum_{k=0}^{T-1} x_c[k]\,h[t-k]
$$

Here, $x_c[t]$ is the binary stimulus vector for condition $c$, $h[t]$ is the HRF, and $\tilde{x}_c[t]$ is the HRF-convolved predictor. The convolution operation places a delayed HRF-shaped response at each time point where the stimulus is active and sums these responses across time.

The Q4 GLM is:

$$
Y_p[t] =
\sum_{c=1}^{6} \beta_{c,p}\,\tilde{x}_c[t] + \epsilon_p[t]
$$

Expanded across the six motor-task conditions:

$$
Y_p[t] =
\beta_{LF,p}(x_{LF} * h)[t] +
\beta_{RF,p}(x_{RF} * h)[t] +
\beta_{LH,p}(x_{LH} * h)[t] +
\beta_{RH,p}(x_{RH} * h)[t] +
\beta_{T,p}(x_T * h)[t] +
\beta_{Cue,p}(x_{Cue} * h)[t] +
\epsilon_p[t]
$$

where $Y_p[t]$ is the measured BOLD signal for brain parcel $p$ at time point $t$, $\beta_{c,p}$ estimates how strongly parcel $p$'s BOLD signal follows condition $c$'s predicted BOLD response, and $\epsilon_p[t]$ is unexplained noise.

For the group level analysis, I focus on the right hand condition. After estimating subject level HRF-aware betas, I test whether the right hand beta is significantly greater than zero across subjects for each parcel. This identifies parcels whose measured BOLD activity reliably follows the HRF-shaped right hand task predictor.
