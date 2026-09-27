# Resultados: modelos e baselines (FR-4)

MAE e RMSE em casos por municipio-semana. `_orig` e a escala de
contagem, `_log` e sobre log1p, que e o que o treino otimiza.


## E1 — recorte 2019-2023, 1.369 municipios, teste 2023

| modelo                  |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   F1_macro |   AUC_macro_ovr |
|:------------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|----------------:|
| baseline: persistencia  |     72557 |     9.4491 |     83.3082 |    0.4003 |    0.5254 |   0.57   |     0.4842 |        nan      |
| baseline: sazonal       |     71188 |    14.0458 |     91.605  |   -0.1038 |    0.7887 |   0.0473 |     0.3236 |        nan      |
| baseline: media_movel_4 |     72557 |    10.891  |     84.5873 |    0.3817 |    0.5705 |   0.5689 |     0.4583 |        nan      |
| XGBoost: e1_sinan       |     72557 |     8.0602 |     85.986  |    0.3611 |    0.4749 |   0.7057 |     0.4011 |          0.7775 |
| XGBoost: e1_sinan_inmet |     72557 |     7.9349 |     84.0184 |    0.39   |    0.4763 |   0.7068 |     0.3992 |          0.7859 |


## E2 — historico completo 2000-2023, 5.565 municipios, teste 2023

| modelo                  |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   F1_macro |   AUC_macro_ovr |
|:------------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|----------------:|
| baseline: persistencia  |    294945 |     4.6535 |     53.3568 |    0.4352 |    0.3387 |   0.5845 |     0.4969 |        nan      |
| baseline: sazonal       |    289380 |     7.6971 |     63.8967 |   -0.0186 |    0.5505 |  -0.0118 |     0.3714 |        nan      |
| baseline: media_movel_4 |    294945 |     5.3826 |     58.2606 |    0.3266 |    0.3676 |   0.588  |     0.4773 |        nan      |
| XGBoost: e2_sinan       |    294945 |     3.7435 |     50.7186 |    0.4897 |    0.3174 |   0.7335 |     0.4514 |          0.8402 |


## US-009 — mesmo recorte agregado por mesorregiao (43 series), teste 2023

| modelo                    |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   F1_macro |   AUC_macro_ovr |
|:--------------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|----------------:|
| baseline: persistencia    |      2279 |    252.099 |     736.004 |    0.3971 |    0.8225 |   0.5322 |     0.5675 |        nan      |
| baseline: sazonal         |      2236 |    377.743 |     891.719 |   -0.1959 |    1.3136 |   0.2166 |     0.3699 |        nan      |
| baseline: media_movel_4   |      2279 |    304.649 |     786.807 |    0.311  |    0.8833 |   0.6294 |     0.5069 |        nan      |
| XGBoost: meso_sinan       |      2279 |    224.024 |     745.502 |    0.3814 |    0.6812 |   0.6924 |     0.4092 |          0.7244 |
| XGBoost: meso_sinan_inmet |      2279 |    195.654 |     678.653 |    0.4874 |    0.6461 |   0.7031 |     0.4336 |          0.7553 |


## US-009 — mesmo recorte agregado por UF (6 series), teste 2023

| modelo                  |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   F1_macro |   AUC_macro_ovr |
|:------------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|----------------:|
| baseline: persistencia  |       318 |    1650.92 |     3501.23 |    0.4544 |    0.7063 |  -0.1518 |     0.5719 |        nan      |
| baseline: sazonal       |       312 |    2361.27 |     4069.1  |    0.1303 |    1.3033 |  -0.4169 |     0.2091 |        nan      |
| baseline: media_movel_4 |       318 |    2020.87 |     3886.46 |    0.3278 |    0.728  |   0.3348 |     0.5244 |        nan      |
| XGBoost: uf_sinan       |       318 |    1204.4  |     2904.63 |    0.6245 |    0.5723 |   0.0872 |     0.3013 |          0.6827 |
| XGBoost: uf_sinan_inmet |       318 |    1305.99 |     3256.91 |    0.5279 |    0.5686 |   0.0754 |     0.2932 |          0.718  |


## Inflacao das metricas pelo vazamento no rotulo (US-002)

Quanto a classificacao parecia melhor quando os limiares de
risco eram calculados sobre o dataset inteiro em vez de so
sobre o treino.

| braco            |   F1_macro_sem_vazamento |   F1_macro_com_vazamento |   inflacao_F1_macro |   AUC_sem_vazamento |   AUC_com_vazamento |   inflacao_AUC |
|:-----------------|-------------------------:|-------------------------:|--------------------:|--------------------:|--------------------:|---------------:|
| e1_sinan         |                   0.4011 |                   0.4284 |              0.0273 |              0.7775 |              0.7978 |         0.0203 |
| e1_sinan_inmet   |                   0.3992 |                   0.4325 |              0.0333 |              0.7859 |              0.8004 |         0.0145 |
| e2_sinan         |                   0.4514 |                   0.4558 |              0.0044 |              0.8402 |              0.8414 |         0.0012 |
| meso_sinan       |                   0.4092 |                   0.4503 |              0.0411 |              0.7244 |              0.7585 |         0.0341 |
| meso_sinan_inmet |                   0.4336 |                   0.4779 |              0.0443 |              0.7553 |              0.7895 |         0.0342 |
| uf_sinan         |                   0.3013 |                   0.3139 |              0.0126 |              0.6827 |              0.7347 |         0.052  |
| uf_sinan_inmet   |                   0.2932 |                   0.4001 |              0.1069 |              0.718  |              0.7918 |         0.0738 |

