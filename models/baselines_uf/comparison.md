# Baselines ingenuos (US-007)

Dataset: `data/model_ready/uf` | teste: [2023] | 318 linhas

MAE e RMSE em casos por municipio-semana; R2_orig na escala de
contagem e R2_log sobre log1p.

| modelo                 |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   acuracia |   F1_macro |   cobertura_teste_pct |
|:-----------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|-----------:|----------------------:|
| baseline_persistencia  |       318 |    1650.92 |     3501.23 |    0.4544 |    0.7063 |  -0.1518 |     0.5818 |     0.5719 |                100    |
| baseline_sazonal       |       312 |    2361.27 |     4069.1  |    0.1303 |    1.3033 |  -0.4169 |     0.2821 |     0.2091 |                 98.11 |
| baseline_media_movel_4 |       318 |    2020.87 |     3886.46 |    0.3278 |    0.728  |   0.3348 |     0.5346 |     0.5244 |                100    |
