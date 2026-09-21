### Test metrics, mean ± SD over seeds (MAE and RMSE in °C)

| run | n_real | n_seeds | mae | rmse | r2 | spearman | val MAE | best epoch |
|---|---|---|---|---|---|---|---|---|
| synthetic_generated | 0 | 5 | 36.8 ± 0.6 | 50.3 ± 0.5 | 0.794 ± 0.004 | 0.904 ± 0.003 | 36.0 | 3.4 |
| synthetic_labeled | 0 | 5 | 38.5 ± 0.7 | 51.4 ± 0.7 | 0.785 ± 0.006 | 0.904 ± 0.002 | 38.5 | 4.0 |
| real_n1000 | 1000 | 5 | 32.3 ± 2.5 | 44.5 ± 2.4 | 0.838 ± 0.018 | 0.923 ± 0.007 | 31.3 | 8.6 |
| real_n250 | 250 | 5 | 39.1 ± 1.4 | 52.3 ± 1.8 | 0.778 ± 0.016 | 0.892 ± 0.007 | 37.2 | 5.4 |
| real_n500 | 500 | 5 | 35.9 ± 1.4 | 48.8 ± 2.0 | 0.806 ± 0.015 | 0.909 ± 0.007 | 34.3 | 5.4 |
| real_nfull | 5297 | 5 | 25.2 ± 1.2 | 35.0 ± 1.3 | 0.900 ± 0.007 | 0.953 ± 0.004 | 24.9 | 6.4 |
| twostage_generated_n1000 | 1000 | 5 | 30.5 ± 1.0 | 42.3 ± 1.2 | 0.854 ± 0.009 | 0.930 ± 0.003 | 29.8 | 4.4 |
| twostage_generated_n250 | 250 | 5 | 34.6 ± 0.6 | 47.2 ± 0.8 | 0.819 ± 0.006 | 0.913 ± 0.003 | 33.2 | 3.6 |
| twostage_generated_n500 | 500 | 5 | 33.2 ± 0.9 | 45.5 ± 1.6 | 0.831 ± 0.012 | 0.920 ± 0.004 | 31.8 | 3.8 |
| twostage_generated_nfull | 5297 | 5 | 24.9 ± 1.0 | 34.5 ± 1.2 | 0.903 ± 0.007 | 0.955 ± 0.004 | 24.5 | 5.4 |
| twostage_labeled_n1000 | 1000 | 5 | 30.2 ± 0.7 | 42.3 ± 0.7 | 0.855 ± 0.005 | 0.931 ± 0.002 | 29.1 | 5.6 |
| twostage_labeled_n250 | 250 | 5 | 34.3 ± 0.7 | 47.2 ± 0.5 | 0.819 ± 0.004 | 0.914 ± 0.004 | 32.8 | 4.2 |
| twostage_labeled_n500 | 500 | 5 | 32.5 ± 0.8 | 45.1 ± 1.5 | 0.834 ± 0.011 | 0.922 ± 0.004 | 31.5 | 3.6 |
| twostage_labeled_nfull | 5297 | 5 | 24.8 ± 2.0 | 35.2 ± 3.6 | 0.898 ± 0.022 | 0.956 ± 0.003 | 24.4 | 6.0 |
| concat_generated_n500 | 500 | 5 | 33.0 ± 0.9 | 45.3 ± 0.8 | 0.833 ± 0.006 | 0.919 ± 0.003 | 31.8 | 5.4 |
| concat_generated_nfull | 5297 | 5 | 24.1 ± 0.7 | 33.8 ± 0.8 | 0.907 ± 0.005 | 0.956 ± 0.002 | 24.0 | 11.0 |
| concat_labeled_n500 | 500 | 5 | 32.8 ± 0.7 | 45.5 ± 1.2 | 0.831 ± 0.009 | 0.921 ± 0.004 | 32.2 | 6.2 |
| concat_labeled_nfull | 5297 | 5 | 26.1 ± 1.3 | 36.5 ± 1.9 | 0.891 ± 0.011 | 0.947 ± 0.007 | 25.3 | 4.6 |
| twostage_selftrain_n500 | 500 | 5 | 34.5 ± 1.0 | 47.5 ± 1.7 | 0.816 ± 0.013 | 0.914 ± 0.005 | 34.1 | 3.0 |
| twostage_shuffled_generated_n500 | 500 | 5 | 54.5 ± 16.7 | 70.4 ± 19.9 | 0.571 ± 0.256 | 0.692 ± 0.307 | 51.3 | 6.6 |
| twostage_shuffled_labeled_n500 | 500 | 5 | 39.5 ± 5.8 | 53.2 ± 7.5 | 0.766 ± 0.066 | 0.880 ± 0.046 | 37.9 | 5.2 |
| synthetic_selftrain | 0 | 5 | 36.2 ± 1.0 | 48.9 ± 1.4 | 0.805 ± 0.011 | 0.908 ± 0.006 | 35.4 | 4.8 |
| synthetic_shuffled_generated | 0 | 5 | 97.1 ± 2.1 | 119.8 ± 3.3 | -0.169 ± 0.064 | -0.111 ± 0.687 | 99.5 | 3.0 |
| synthetic_shuffled_labeled | 0 | 5 | 105.0 ± 4.9 | 131.9 ± 5.9 | -0.417 ± 0.128 | 0.115 ± 0.528 | 108.5 | 3.0 |
| rf_concat_generated_n500 | 500 | 5 | 39.9 ± 0.3 | 55.8 ± 0.8 | 0.746 ± 0.008 | 0.875 ± 0.005 |  |  |
| rf_concat_generated_nfull | 5297 | 5 | 27.3 ± 0.1 | 39.2 ± 0.1 | 0.875 ± 0.001 | 0.938 ± 0.000 |  |  |
| rf_concat_labeled_n500 | 500 | 5 | 40.3 ± 0.4 | 54.5 ± 0.8 | 0.759 ± 0.007 | 0.882 ± 0.003 |  |  |
| rf_concat_labeled_nfull | 5297 | 5 | 27.6 ± 0.0 | 39.6 ± 0.1 | 0.872 ± 0.000 | 0.934 ± 0.000 |  |  |
| rf_real_n500 | 500 | 5 | 40.4 ± 0.5 | 54.5 ± 1.3 | 0.759 ± 0.011 | 0.873 ± 0.007 |  |  |
| rf_real_nfull | 5297 | 5 | 26.5 ± 0.1 | 37.8 ± 0.1 | 0.884 ± 0.001 | 0.942 ± 0.000 |  |  |
| rf_synthetic_generated | 0 | 5 | 47.5 ± 0.1 | 65.4 ± 0.1 | 0.652 ± 0.001 | 0.837 ± 0.000 |  |  |
| rf_synthetic_labeled | 0 | 5 | 47.1 ± 0.0 | 62.1 ± 0.1 | 0.686 ± 0.001 | 0.858 ± 0.000 |  |  |

### Test MAE (°C) per seed

| run | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| concat_generated_n500 | 32.59 | 33.07 | 32.93 | 32.1 | 34.39 |
| concat_generated_nfull | 23.1 | 23.9 | 24.97 | 24.29 | 24.24 |
| concat_labeled_n500 | 32.45 | 33.72 | 32.75 | 33.29 | 31.89 |
| concat_labeled_nfull | 26.42 | 27.29 | 25.1 | 27.2 | 24.41 |
| real_n1000 | 30.24 | 32.05 | 36.58 | 30.5 | 32.38 |
| real_n250 | 41.35 | 37.6 | 38.23 | 39.4 | 38.89 |
| real_n500 | 35.46 | 35.84 | 37.73 | 36.33 | 33.97 |
| real_nfull | 24.01 | 26.45 | 24.06 | 26.35 | 25.29 |
| rf_concat_generated_n500 | 39.54 | 39.88 | 40.3 | 40.28 | 39.75 |
| rf_concat_generated_nfull | 27.27 | 27.43 | 27.29 | 27.31 | 27.27 |
| rf_concat_labeled_n500 | 40.44 | 41.05 | 39.94 | 40.06 | 40.13 |
| rf_concat_labeled_nfull | 27.57 | 27.64 | 27.58 | 27.58 | 27.59 |
| rf_real_n500 | 41.2 | 40.63 | 40.37 | 40.09 | 39.8 |
| rf_real_nfull | 26.53 | 26.49 | 26.51 | 26.47 | 26.66 |
| rf_synthetic_generated | 47.49 | 47.53 | 47.51 | 47.56 | 47.31 |
| rf_synthetic_labeled | 47.03 | 47.11 | 47.04 | 47.03 | 47.08 |
| synthetic_generated | 37.71 | 36.53 | 36.16 | 36.74 | 37.04 |
| synthetic_labeled | 39.45 | 38.06 | 38.91 | 37.78 | 38.29 |
| synthetic_selftrain | 35.44 | 36.41 | 37.8 | 36.23 | 35.32 |
| synthetic_shuffled_generated | 98.88 | 93.85 | 98.42 | 98.07 | 96.16 |
| synthetic_shuffled_labeled | 99.11 | 110.66 | 109.5 | 103.26 | 102.57 |
| twostage_generated_n1000 | 30.14 | 32.37 | 29.9 | 30.16 | 30.01 |
| twostage_generated_n250 | 34.44 | 34.67 | 33.66 | 35.44 | 34.79 |
| twostage_generated_n500 | 34.31 | 33.63 | 33.26 | 32.48 | 32.07 |
| twostage_generated_nfull | 24.22 | 25.93 | 23.76 | 25.92 | 24.69 |
| twostage_labeled_n1000 | 29.87 | 30.92 | 30.82 | 29.32 | 30.2 |
| twostage_labeled_n250 | 33.83 | 33.66 | 34.64 | 35.48 | 34.08 |
| twostage_labeled_n500 | 32.78 | 33.5 | 31.67 | 32.75 | 31.76 |
| twostage_labeled_nfull | 23.66 | 23.54 | 23.17 | 25.76 | 27.82 |
| twostage_selftrain_n500 | 35.07 | 34.74 | 34.03 | 35.65 | 33.05 |
| twostage_shuffled_generated_n500 | 60.04 | 42.26 | 46.7 | 81.29 | 42.15 |
| twostage_shuffled_labeled_n500 | 44.05 | 35.95 | 35.73 | 34.34 | 47.41 |

### Best epoch (early stopping on the validation MAE) per seed

| run | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| concat_generated_n500 | 8.0 | 5.0 | 5.0 | 6.0 | 3.0 |
| concat_generated_nfull | 16.0 | 10.0 | 5.0 | 10.0 | 14.0 |
| concat_labeled_n500 | 7.0 | 7.0 | 2.0 | 2.0 | 13.0 |
| concat_labeled_nfull | 3.0 | 3.0 | 4.0 | 3.0 | 10.0 |
| real_n1000 | 8.0 | 9.0 | 2.0 | 12.0 | 12.0 |
| real_n250 | 3.0 | 6.0 | 4.0 | 9.0 | 5.0 |
| real_n500 | 6.0 | 7.0 | 2.0 | 4.0 | 8.0 |
| real_nfull | 8.0 | 3.0 | 9.0 | 3.0 | 9.0 |
| synthetic_generated | 2.0 | 2.0 | 6.0 | 2.0 | 5.0 |
| synthetic_labeled | 5.0 | 6.0 | 2.0 | 3.0 | 4.0 |
| synthetic_selftrain | 10.0 | 3.0 | 6.0 | 2.0 | 3.0 |
| synthetic_shuffled_generated | 5.0 | 3.0 | 1.0 | 4.0 | 2.0 |
| synthetic_shuffled_labeled | 2.0 | 2.0 | 5.0 | 4.0 | 2.0 |
| twostage_generated_n1000 | 3.0 | 3.0 | 6.0 | 6.0 | 4.0 |
| twostage_generated_n250 | 3.0 | 4.0 | 1.0 | 6.0 | 4.0 |
| twostage_generated_n500 | 2.0 | 6.0 | 2.0 | 6.0 | 3.0 |
| twostage_generated_nfull | 8.0 | 3.0 | 7.0 | 3.0 | 6.0 |
| twostage_labeled_n1000 | 3.0 | 9.0 | 6.0 | 6.0 | 4.0 |
| twostage_labeled_n250 | 4.0 | 8.0 | 1.0 | 4.0 | 4.0 |
| twostage_labeled_n500 | 2.0 | 4.0 | 5.0 | 4.0 | 3.0 |
| twostage_labeled_nfull | 8.0 | 6.0 | 9.0 | 3.0 | 4.0 |
| twostage_selftrain_n500 | 3.0 | 3.0 | 5.0 | 1.0 | 3.0 |
| twostage_shuffled_generated_n500 | 5.0 | 8.0 | 8.0 | 7.0 | 5.0 |
| twostage_shuffled_labeled_n500 | 6.0 | 6.0 | 5.0 | 4.0 | 5.0 |

### Seed-matched differences in test MAE (°C); negative = the first configuration is better. With 5 pairs the Wilcoxon p-value cannot be below 0.0625

| comparison (first minus second) | seeds | dMAE mean | SD | min | max | same sign | paired t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| two-stage generated vs real only, n250 | 5 | -4.49 | 1.47 | -6.9 | -2.93 | 5/5 | 0.0024 | 0.0625 |
| two-stage labeled vs real only, n250 | 5 | -4.75 | 1.61 | -7.52 | -3.59 | 5/5 | 0.0027 | 0.0625 |
| two-stage generated vs real only, n500 | 5 | -2.72 | 1.39 | -4.47 | -1.15 | 5/5 | 0.012 | 0.0625 |
| two-stage labeled vs real only, n500 | 5 | -3.37 | 1.6 | -6.06 | -2.21 | 5/5 | 0.0091 | 0.0625 |
| two-stage generated vs real only, n1000 | 5 | -1.83 | 2.9 | -6.68 | 0.32 | 4/5 | 0.2306 | 0.1875 |
| two-stage labeled vs real only, n1000 | 5 | -2.12 | 2.14 | -5.76 | -0.36 | 5/5 | 0.0905 | 0.0625 |
| two-stage generated vs real only, nfull | 5 | -0.33 | 0.32 | -0.6 | 0.21 | 4/5 | 0.0851 | 0.125 |
| two-stage labeled vs real only, nfull | 5 | -0.44 | 1.95 | -2.91 | 2.53 | 4/5 | 0.6387 | 0.4375 |
| two-stage vs concatenation, generated, n500 | 5 | 0.14 | 1.48 | -2.31 | 1.73 | 4/5 | 0.8463 | 0.625 |
| concatenation generated vs real only, n500 | 5 | -2.85 | 2.03 | -4.8 | 0.42 | 4/5 | 0.0346 | 0.125 |
| two-stage vs concatenation, labeled, n500 | 5 | -0.33 | 0.53 | -1.08 | 0.33 | 4/5 | 0.2344 | 0.3125 |
| concatenation labeled vs real only, n500 | 5 | -3.05 | 1.18 | -4.98 | -2.08 | 5/5 | 0.0044 | 0.0625 |
| two-stage vs concatenation, generated, nfull | 5 | 0.8 | 1.27 | -1.21 | 2.03 | 4/5 | 0.2306 | 0.3125 |
| concatenation generated vs real only, nfull | 5 | -1.13 | 1.33 | -2.56 | 0.91 | 4/5 | 0.1311 | 0.1875 |
| two-stage vs concatenation, labeled, nfull | 5 | -1.3 | 2.77 | -3.75 | 3.4 | 4/5 | 0.3544 | 0.4375 |
| concatenation labeled vs real only, nfull | 5 | 0.85 | 1.17 | -0.87 | 2.41 | 4/5 | 0.1771 | 0.3125 |
| self-training vs LLM-labeled (two-stage, n500) | 5 | 2.02 | 0.73 | 1.24 | 2.91 | 5/5 | 0.0034 | 0.0625 |
| self-training vs real only, n500 | 5 | -1.36 | 1.34 | -3.7 | -0.39 | 5/5 | 0.0859 | 0.0625 |
| shuffled vs unshuffled generated (two-stage, n500) | 5 | 21.34 | 16.77 | 8.63 | 48.82 | 5/5 | 0.0466 | 0.0625 |
| shuffled vs unshuffled labeled (two-stage, n500) | 5 | 7.01 | 6.16 | 1.6 | 15.65 | 5/5 | 0.0637 | 0.0625 |
| shuffled generated vs real only, n500 | 5 | 18.62 | 16.43 | 6.42 | 44.96 | 5/5 | 0.0644 | 0.0625 |
| shuffled labeled vs real only, n500 | 5 | 3.63 | 7.01 | -2.0 | 13.44 | 3/5 | 0.3111 | 0.625 |
| labeled only vs generated only (stage 1) | 5 | 1.66 | 0.67 | 1.04 | 2.75 | 5/5 | 0.0051 | 0.0625 |
| random forest: real + generated vs real only, n500 | 5 | -0.47 | 0.75 | -1.66 | 0.19 | 4/5 | 0.2365 | 0.3125 |
| random forest: real + labeled vs real only, n500 | 5 | -0.09 | 0.5 | -0.75 | 0.42 | 3/5 | 0.7054 | 0.625 |
| random forest: real + generated vs real only, nfull | 5 | 0.78 | 0.12 | 0.62 | 0.94 | 5/5 | 0.0001 | 0.0625 |
| random forest: real + labeled vs real only, nfull | 5 | 1.06 | 0.08 | 0.93 | 1.15 | 5/5 | 0.0 | 0.0625 |

### Stage-1 (synthetic-only) students against the zero-shot teacher on the same test polymers (paired bootstrap, 1,000 resamples)

| student | seed | student MAE - teacher MAE | CI 2.5 % | CI 97.5 % | CI excludes 0 |
|---|---|---|---|---|---|
| synthetic_generated | 42 | 0.4 | -1.13 | 2.01 | False |
| synthetic_generated | 43 | -0.78 | -2.24 | 0.65 | False |
| synthetic_generated | 44 | -1.15 | -2.5 | 0.33 | False |
| synthetic_generated | 45 | -0.57 | -1.96 | 0.96 | False |
| synthetic_generated | 46 | -0.27 | -1.72 | 1.08 | False |
| synthetic_labeled | 42 | 2.15 | 0.97 | 3.3 | True |
| synthetic_labeled | 43 | 0.75 | -0.39 | 2.01 | False |
| synthetic_labeled | 44 | 1.61 | 0.45 | 2.83 | True |
| synthetic_labeled | 45 | 0.47 | -0.59 | 1.55 | False |
| synthetic_labeled | 46 | 0.98 | -0.04 | 2.13 | False |

### Experimental noise floor and model error on the test polymers with repeated measurements (°C)

| quantity | value |
|---|---|
| test polymers with 2+ measurements | 54.0 |
| median experimental SD | 5.74 |
| mean experimental SD | 17.27 |
| expected |difference| of two measurements, from the median SD (1.128 x SD) | 6.48 |
| expected |difference| of two measurements, from the mean SD | 19.49 |
| MAE on these polymers: zero-shot teacher | 25.95 |
| MAE on these polymers: real only, full | 20.9 |
| MAE on these polymers: real only, n500 | 34.09 |
| MAE on these polymers: generated only | 29.61 |
| MAE on these polymers: labeled only | 29.82 |
| MAE on these polymers: two-stage generated, n500 | 29.81 |
| MAE on these polymers: two-stage labeled, n500 | 30.85 |
| MAE on these polymers: two-stage generated, full | 21.57 |
| MAE on these polymers: two-stage labeled, full | 23.66 |
| MAE on these polymers: random forest, real full | 21.32 |
| MAE on the other 1417 test polymers: zero-shot teacher | 37.74 |
| MAE on the other 1417 test polymers: real only, full | 25.4 |
| MAE on the other 1417 test polymers: real only, n500 | 35.93 |
| MAE on the other 1417 test polymers: generated only | 37.11 |
| MAE on the other 1417 test polymers: labeled only | 38.83 |
| MAE on the other 1417 test polymers: two-stage generated, n500 | 33.28 |
| MAE on the other 1417 test polymers: two-stage labeled, n500 | 32.55 |
| MAE on the other 1417 test polymers: two-stage generated, full | 25.03 |
| MAE on the other 1417 test polymers: two-stage labeled, full | 24.83 |
| MAE on the other 1417 test polymers: random forest, real full | 26.73 |

### Test MAE (°C) by reliability flag of the dataset (black = one data point)

| reliability | n | zero-shot teacher | real only, full | real only, n500 | generated only | labeled only | two-stage generated, n500 | two-stage labeled, n500 | two-stage generated, full | two-stage labeled, full | random forest, real full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| black | 1418.0 | 37.7 | 25.4 | 35.9 | 37.1 | 38.8 | 33.3 | 32.5 | 25.0 | 24.8 | 26.7 |
| black/red | 1.0 | 89.1 | 34.0 | 46.4 | 66.3 | 97.5 | 32.3 | 54.8 | 35.9 | 33.9 | 35.5 |
| gold | 23.0 | 25.6 | 21.5 | 30.3 | 28.2 | 29.3 | 29.3 | 29.5 | 22.7 | 23.3 | 20.6 |
| red | 1.0 | 3.9 | 30.8 | 41.6 | 24.7 | 32.3 | 33.6 | 22.6 | 28.2 | 18.7 | 12.9 |
| yellow | 28.0 | 25.3 | 20.0 | 37.1 | 29.9 | 27.7 | 29.9 | 32.0 | 20.5 | 24.3 | 22.2 |
| all | 1471.0 | 37.3 | 25.2 | 35.9 | 36.8 | 38.5 | 33.2 | 32.5 | 24.9 | 24.8 | 26.5 |

### Test MAE (°C) by band of the measured Tg

| tg_bin | n | zero-shot teacher | real only, full | real only, n500 | generated only | labeled only | two-stage generated, n500 | two-stage labeled, n500 | two-stage generated, full | two-stage labeled, full | random forest, real full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [-150, 0) | 143.0 | 31.1 | 23.5 | 33.9 | 27.3 | 32.1 | 30.8 | 31.9 | 23.8 | 26.0 | 27.8 |
| [0, 100) | 431.0 | 30.6 | 23.5 | 31.0 | 30.6 | 30.9 | 27.6 | 27.4 | 23.1 | 22.6 | 23.4 |
| [100, 200) | 411.0 | 42.3 | 26.9 | 37.3 | 37.3 | 43.2 | 35.3 | 35.5 | 26.6 | 25.8 | 27.7 |
| [200, 300) | 361.0 | 37.6 | 22.6 | 33.6 | 35.1 | 39.6 | 31.7 | 30.3 | 22.1 | 22.2 | 25.1 |
| [300, 501) | 125.0 | 50.3 | 35.7 | 56.8 | 72.8 | 53.3 | 52.0 | 47.3 | 34.9 | 35.1 | 36.2 |
| all | 1471.0 | 37.3 | 25.2 | 35.9 | 36.8 | 38.5 | 33.2 | 32.5 | 24.9 | 24.8 | 26.5 |

### Test MAE (°C) by maximum Tanimoto similarity of the test polymer to the real training set

| similarity_bin | n | zero-shot teacher | real only, full | real only, n500 | generated only | labeled only | two-stage generated, n500 | two-stage labeled, n500 | two-stage generated, full | two-stage labeled, full | random forest, real full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [0.0, 0.4) | 34.0 | 42.2 | 38.6 | 47.1 | 44.0 | 38.6 | 44.9 | 43.7 | 39.7 | 41.7 | 51.6 |
| [0.4, 0.55) | 135.0 | 43.1 | 33.8 | 45.5 | 41.0 | 44.4 | 40.6 | 41.5 | 32.9 | 33.8 | 40.0 |
| [0.55, 0.7) | 281.0 | 40.9 | 28.5 | 39.4 | 39.6 | 40.9 | 36.8 | 36.5 | 28.1 | 28.3 | 34.0 |
| [0.7, 0.85) | 413.0 | 40.7 | 25.7 | 37.5 | 39.5 | 40.4 | 35.0 | 34.3 | 24.9 | 24.6 | 26.4 |
| [0.85, 1.0) | 608.0 | 31.7 | 20.7 | 30.3 | 32.4 | 34.8 | 27.9 | 26.8 | 20.8 | 20.3 | 18.8 |
| all | 1471.0 | 37.3 | 25.2 | 35.9 | 36.8 | 38.5 | 33.2 | 32.5 | 24.9 | 24.8 | 26.5 |

### Sensitivity: test polymers whose repeated measurements disagree by SD > 30 °C, against all others

| high_sd | n | zero-shot teacher | real only, full | real only, n500 | generated only | labeled only | two-stage generated, n500 | two-stage labeled, n500 | two-stage generated, full | two-stage labeled, full | random forest, real full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SD > 30 C | 7.0 | 43.0 | 34.2 | 56.2 | 46.8 | 51.7 | 43.5 | 40.4 | 34.5 | 37.9 | 33.9 |
| other | 1464.0 | 37.3 | 25.2 | 35.8 | 36.8 | 38.4 | 33.1 | 32.5 | 24.9 | 24.7 | 26.5 |
| all | 1471.0 | 37.3 | 25.2 | 35.9 | 36.8 | 38.5 | 33.2 | 32.5 | 24.9 | 24.8 | 26.5 |

### Patience ablation: test MAE (°C) of the real-only and two-stage runs when early stopping waits 3, 5 or 10 epochs; dMAE = two-stage minus real only, seed-matched

| n_real | patience | seeds | best epoch, real only | MAE real | SD real | MAE generated | SD generated | MAE labeled | SD labeled | dMAE generated | dSD generated | same sign generated | paired t p generated | dMAE labeled | dSD labeled | same sign labeled | paired t p labeled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 3 | 5 | 5.4 | 38.992 | 1.382 | 34.632 | 0.531 | 34.39 | 0.993 | -4.361 | 1.475 | 5/5 | 0.003 | -4.602 | 1.569 | 5/5 | 0.003 |
| 250 | 5 | 5 | 9.8 | 38.151 | 0.901 | 34.461 | 0.618 | 34.39 | 0.993 | -3.689 | 0.921 | 5/5 | 0.001 | -3.761 | 0.578 | 5/5 | 0.0 |
| 250 | 10 | 5 | 23.6 | 37.748 | 0.688 | 34.375 | 0.466 | 33.885 | 0.593 | -3.373 | 0.918 | 5/5 | 0.001 | -3.864 | 0.861 | 5/5 | 0.001 |
| 500 | 3 | 5 | 5.4 | 35.845 | 1.526 | 32.98 | 0.706 | 32.401 | 0.671 | -2.865 | 1.25 | 5/5 | 0.007 | -3.444 | 1.544 | 5/5 | 0.008 |
| 500 | 5 | 5 | 9.2 | 35.44 | 1.531 | 32.724 | 0.705 | 32.188 | 0.696 | -2.715 | 1.218 | 5/5 | 0.008 | -3.252 | 1.559 | 5/5 | 0.01 |
| 500 | 10 | 5 | 20.0 | 34.069 | 0.898 | 32.566 | 0.907 | 31.957 | 0.736 | -1.503 | 1.251 | 4/5 | 0.055 | -2.112 | 0.954 | 5/5 | 0.008 |
