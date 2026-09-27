# Baselines ingenuos (US-007)

Dataset: `data/model_ready/mesorregiao` | teste: [2023] | 2,279 linhas

MAE e RMSE em casos por municipio-semana; R2_orig na escala de
contagem e R2_log sobre log1p.

| modelo                 |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   acuracia |   F1_macro |   cobertura_teste_pct |
|:-----------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|-----------:|----------------------:|
| baseline_persistencia  |      2279 |    252.099 |     736.004 |    0.3971 |    0.8225 |   0.5322 |     0.602  |     0.5675 |                100    |
| baseline_sazonal       |      2236 |    377.743 |     891.719 |   -0.1959 |    1.3136 |   0.2166 |     0.4495 |     0.3699 |                 98.11 |
| baseline_media_movel_4 |      2279 |    304.649 |     786.807 |    0.311  |    0.8833 |   0.6294 |     0.545  |     0.5069 |                100    |
