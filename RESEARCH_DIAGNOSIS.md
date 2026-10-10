# MI-BCI reproducibility diagnosis (2026-10-10)

This document separates observations in the saved results from claims that
require a new experiment. It does not overwrite the historical v4 outputs or
the in-progress v5 Colab notebook.

## Decision-critical problems

| Priority | Observation | Consequence | Resolution and status |
|---|---|---|---|
| 1 | `src/ablation_study.py` and `src/ablation_study_v5.py` select the best epoch with the outer LOSO **test subject's** macro-F1. The S3 notebook follows the same pattern. | The subject is not held out from model selection. Absolute test accuracy and paired comparisons may be optimistic. Equal treatment in v4/v5 does not guarantee equal bias. | New subject-disjoint runner: `src/rigorous_loso.py`. It uses one inner validation subject and reads the test subject once after model selection. Dry-run verified; full training is pending GPU execution. |
| 2 | GigaDB flags 724 movement-related and 30 voltage-related trials among the 51 available raw `.mat` files. Their union is **753**, because one trial has both flags. The original v4 training retained them. | sEMG discriminability and gate-weight claims can reflect actual movement. Removing entire subjects also changes the study population. | Compare `dirty_train` and `clean_train` on the **same clean outer test trials**, with the same clean validation subject and seed. Both conditions must run under the new protocol. The v5 run in progress answers a related question but retains test-fold early stopping. |
| 3 | `S1_S2_Preprocessing_MemberA.ipynb` normalizes each subject across all epochs **before** v5 removes flagged trials from HDF5. ICA was also fitted to the complete continuous recording. | v5 is a trial-selection analysis, not a fully clean preprocessing run. Removed trials can still influence preprocessing estimates. | The new runner re-fits EEG channel z-score after exclusion for every clean dataset, removing the flagged trials' contribution to those statistics. Rebuilding ICA/filtering from raw signals is still pending. `s06.mat` must be restored or the same 51-subject pool used in both conditions. |
| 4 | Raw `s06.mat` is absent while its v4 HDF5 exists. v5 evaluates 49 eligible subjects after also excluding s29/s34 from evaluation. | A 52-subject v4 mean and 49-subject clean mean do not estimate a paired change. The two most contaminated subjects leave the evaluation set. | Restrict all comparisons to identical subjects and identical clean test trials. Report s29/s34 separately as non-estimable under the chosen minimum-trial rule. The new runner uses the same 51-subject pool and 49 evaluation subjects for both conditions. |
| 5 | The JNE submission manuscript's conditional-benefit and gate-mechanism claims rely on post-hoc associations. The gate/SNR correlation is null. Movement-contamination rate correlates with sEMG-only accuracy and gate weight. | The current files cannot establish that the gate responds to useful covert MI rather than movement. | Treat all gate correlations as exploratory. After paired retraining, test a predeclared clean-cohort association and a fixed-versus-adaptive gate ablation. Do not claim a causal gate mechanism from weights alone. |

## Additional corrections

- `BCI_Research/results/movement_flagged_trials_52.csv` had `bad_tot=11` for
  s15 even though the verified union is 10. The derived CSV is corrected here;
  the authoritative per-epoch list remains
  `BCI_Research/results/ablation_v5_clean/excluded_epochs.json`.
- The existing cross-dataset run reports 49.8% on BCI IV 2a. Its script mapped
  the final GDF channel to Oz rather than POz and normalized each **trial**
  separately while the GigaDB HDF5 was normalized per **subject**. Both code
  paths are corrected in `src/transfer_bcic2a.py`. The old 49.8% is a historical
  result; it must not be presented as the result of the corrected pipeline.
  Re-run after the local MNE/GDF reader issue is resolved and verify label,
  timing, and channel mapping from the dataset metadata.
  In this workspace MNE 1.7.0 with NumPy 2.0.2 raises `OverflowError` while
  parsing `A01T.gdf`, so the corrected transfer score could not be measured.
- `BCI_Research/results/comparison_table.csv` contains a 69.7% concat entry
  labeled as literature; the repository has no concat training branch or
  per-subject output. It cannot serve as an internal attention ablation.
- The VR s03 result is HDF5 replay. The 97.89% accuracy is 186/190 predictions
  after 10 confidence skips. The 30.94 ms figure is server-side processing;
  the session summary has no measured acquisition-to-avatar latency.
- ERD, Grad-CAM, and SHAP artifacts exist. In the saved attribution CSVs, Oz is
  the most frequent top EEG channel (67/104 Grad-CAM and 76/104 SHAP rows).
  These artifacts need visual-cue/timing and attribution-method checks before
  being cited as evidence of a sensorimotor decision mechanism.

## New paired experiment

Run on a CUDA machine after copying this code and the 51-subject exclusion
manifest. Use a new output directory; do not write into `ablation_v5_clean`.
The commands below are examples for one seed and one model condition:

```bash
python src/rigorous_loso.py --condition dirty_train --model_type fusion --seed 42
python src/rigorous_loso.py --condition clean_train --model_type fusion --seed 42
```

Repeat for `eeg_only`, `emg_only`, and at least seeds 42, 1337, 2024. The
runner stores a configuration manifest next to each resumable CSV and rejects
incompatible resumes. The two conditions share test/validation subject IDs and
are both scored on clean test trials. Its input still inherits the HDF5
ICA and filtering above. It re-normalizes retained EEG epochs per subject;
the resulting comparison tests trial removal plus clean-only normalization
**given those saved post-ICA HDF5 features**, not the entire raw-data effect.

After all 18 runs (2 training conditions × 3 models × 3 seeds) complete:

```bash
python src/analyze_rigorous_loso.py
```

The analysis refuses incomplete or unpaired runs, averages seeds **within each
subject**, then reports paired effects, subject-bootstrap intervals, and
Wilcoxon p-values. It also reports whether cleaning changes fusion's advantage
over EEG-only, the decision-critical contrast.

The primary paired outcome should be per-subject fusion-minus-EEG accuracy
and Cohen's kappa, comparing clean versus dirty training with confidence
intervals and the full per-subject distribution. Report all 49 subjects,
including negative gains. Gate-weight correlations, subgroup boundaries, and
right-bias analyses should be declared exploratory unless fixed in advance.

### Faster subject-disjoint option: fixed five-fold evaluation

`notebooks/S7_Rigorous_Group5_Colab.ipynb` runs a separate five-fold protocol
using `src/rigorous_group5.py`. Its fixed outer partition tests each of the 49
eligible subjects once per seed. A clean validation subject is drawn only from
the training side of each fold. The 2 conditions × 3 models × 3 seeds require
90 model fits instead of LOSO's 882; each fit evaluates 9 or 10 test subjects.
Results are stored in `results/rigorous_group5`, isolated from LOSO results.
The runner saves completed folds and resumes compatible runs. It uses the same
HDF5 trial selection, clean EEG re-normalization, and HDF5 streaming loader as
the LOSO runner. The five-fold design trains on 40 or 41 other subjects per fit
rather than 49, so absolute scores cannot be compared directly with LOSO.

`src/analyze_rigorous_group5.py` checks all 18 complete paired runs and uses
whole-fold cluster resampling for intervals: subjects tested by one fitted
model are not independent model replications. Only five outer clusters are
available, so intervals are exploratory and should not be used as a definitive
significance claim. The main contrast remains the clean-versus-dirty change
in fusion's gain over EEG-only. The notebook does not start training by itself;
the user must execute its cells with a Colab GPU kernel.

## Execution in this workspace

- Verified: 52 HDF5 subjects, 51 raw `.mat` subjects, 51 raw subjects in the
  exclusion manifest, 49 eligible for evaluation, and 753 union exclusions.
- Verified: v4 accuracy means 72.154% EEG, 65.003% sEMG, 74.178% fusion;
  all 51 available fusion checkpoints reproduce their saved full-trial score.
- Verified: secondary clean-only evaluation is 73.137% versus 73.985% on all
  trials for those 51 checkpoints. This does **not** test clean retraining.
- On the 49 eligible evaluation subjects, the same secondary contrast is
  73.164% clean versus 73.628% full, a 0.464 percentage-point difference
  (paired Wilcoxon p=0.0619). The 51-subject contrast is 0.849 percentage
  points (p=0.0228). This change matters because s29/s34 have the highest
  exclusion rates and cannot support a balanced clean evaluation.
- The latest saved Colab output shows 49/49 sEMG folds and 37/49 EEG folds for
  v5 seed 42, with no completed fusion output in the local folder.
- No CUDA device is available locally. A full 49-fold × conditions × seeds
  training run has not been performed here. The new runner's subject splits
  and trial counts were checked by dry-run.
