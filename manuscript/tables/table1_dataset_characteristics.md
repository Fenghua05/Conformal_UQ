# Table 1 | Dataset characteristics

| Dataset | Domain | N | Features (raw) | Features (post-transform, max) | Minority label | Minority ratio | Train (maj/min) | Cal. pool (maj/min) | Test (maj/min) |
|---|---|---:|---:|---:|---|---:|---|---|---|
| kr-vs-kp (`openml_3_kr_vs_kp`) | chess endgame | 3,196 | 36 (0 num / 36 cat) | 73 | nowin | 0.4778 | 1001 / 916 | 334 / 305 | 334 / 306 |
| mushroom (`openml_24_mushroom`) | mushroom morphology | 8,124 | 22 (0 num / 22 cat) | 117 | p | 0.4820 | 2524 / 2350 | 842 / 783 | 842 / 783 |
| nomao (`openml_1486_nomao`) | entity matching | 34,465 | 118 (89 num / 29 cat) | 174 | 1 | 0.2856 | 14773 / 5906 | 4924 / 1969 | 4924 / 1969 |
| phoneme (`openml_1489_phoneme`) | speech | 5,404 | 5 (5 num / 0 cat) | 5 | 2 | 0.2935 | 2290 / 952 | 764 / 317 | 764 / 317 |
| adult (`openml_1590_adult`) | census income | 48,842 | 14 (6 num / 8 cat) | 108 | >50K | 0.2393 | 22292 / 7012 | 7432 / 2337 | 7431 / 2338 |
| PhishingWebsites (`openml_4534_phishingwebsite`) | web security | 11,055 | 30 (0 num / 30 cat) | 68 | -1 | 0.4431 | 3694 / 2939 | 1232 / 979 | 1231 / 980 |
| higgs (`openml_23512_higgs`) | particle-physics simulation | 98,050 | 28 (28 num / 0 cat) | 28 | 0 | 0.4714 | 31096 / 27734 | 10366 / 9244 | 10365 / 9245 |
| numerai28.6 (`openml_23517_numerai28_6`) | financial benchmark | 96,320 | 21 (21 num / 0 cat) | 21 | 0 | 0.4948 | 29195 / 28597 | 9731 / 9533 | 9732 / 9532 |

Class counts are means over the 10 frozen seeds of the locked v1.1 splits; per-seed min/max values are in the CSV. Features (post-transform, max) is the registry maximum over seeds after train-only one-hot encoding. Minority ratio = minority class count / N in the raw dataset (registry-defined minority label).
