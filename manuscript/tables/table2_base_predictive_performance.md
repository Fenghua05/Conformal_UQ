# Table 2 | Base predictive performance

AUROC / AUPRC on the 20% test split; mean (SD) across the 10 frozen seeds per dataset x predictive pipeline.

| Dataset | AUROC: Logistic Regression | AUROC: XGBoost | AUROC: TabPFN | AUPRC: Logistic Regression | AUPRC: XGBoost | AUPRC: TabPFN |
|---|---:|---:|---:|---:|---:|---:|
| kr-vs-kp | 0.994 (0.002) | 0.999 (0.001) | 1.000 (0.000) | 0.993 (0.002) | 0.999 (0.001) | 1.000 (0.000) |
| mushroom | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) | 1.000 (0.000) |
| nomao | 0.988 (0.001) | 0.995 (0.001) | 0.996 (0.000) | 0.972 (0.002) | 0.989 (0.001) | 0.991 (0.001) |
| phoneme | 0.812 (0.009) | 0.951 (0.003) | 0.972 (0.003) | 0.587 (0.022) | 0.891 (0.008) | 0.938 (0.006) |
| adult | 0.904 (0.002) | 0.928 (0.002) | 0.916 (0.002) | 0.760 (0.007) | 0.828 (0.005) | 0.793 (0.005) |
| PhishingWebsites | 0.986 (0.001) | 0.995 (0.001) | 0.998 (0.001) | 0.984 (0.001) | 0.995 (0.001) | 0.997 (0.001) |
| higgs | 0.681 (0.004) | 0.805 (0.003) | 0.822 (0.003) | 0.661 (0.004) | 0.786 (0.004) | 0.806 (0.003) |
| numerai28.6 | 0.530 (0.002) | 0.519 (0.002) | 0.531 (0.003) | 0.521 (0.003) | 0.513 (0.002) | 0.524 (0.003) |

Base predictive performance is invariant across CP method and m by construction (the same base probabilities feed every CP cell); this invariance is verified in the Stage 12 QA evidence.
