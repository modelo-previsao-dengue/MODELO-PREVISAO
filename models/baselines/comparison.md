# Baselines ingenuos (US-007)

Dataset: `data/model_ready` | teste: [2023] | 72,557 linhas

MAE e RMSE em casos por municipio-semana; R2_orig na escala de
contagem e R2_log sobre log1p.

| modelo                 |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   acuracia |   F1_macro |   cobertura_teste_pct |
|:-----------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|-----------:|----------------------:|
| baseline_persistencia  |     72557 |     9.4491 |     83.3082 |    0.4003 |    0.5254 |   0.57   |     0.669  |     0.4842 |                100    |
| baseline_sazonal       |     71188 |    14.0458 |     91.605  |   -0.1038 |    0.7887 |   0.0473 |     0.5754 |     0.3236 |                 98.11 |
| baseline_media_movel_4 |     72557 |    10.891  |     84.5873 |    0.3817 |    0.5705 |   0.5689 |     0.5843 |     0.4583 |                100    |
