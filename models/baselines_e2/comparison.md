# Baselines ingenuos (US-007)

Dataset: `data/model_ready/e2` | teste: [2023] | 294,945 linhas

MAE e RMSE em casos por municipio-semana; R2_orig na escala de
contagem e R2_log sobre log1p.

| modelo                 |   n_teste |   MAE_orig |   RMSE_orig |   R2_orig |   MAE_log |   R2_log |   acuracia |   F1_macro |   cobertura_teste_pct |
|:-----------------------|----------:|-----------:|------------:|----------:|----------:|---------:|-----------:|-----------:|----------------------:|
| baseline_persistencia  |    294945 |     4.6535 |     53.3568 |    0.4352 |    0.3387 |   0.5845 |     0.7476 |     0.4969 |                100    |
| baseline_sazonal       |    289380 |     7.6971 |     63.8967 |   -0.0186 |    0.5505 |  -0.0118 |     0.6585 |     0.3714 |                 98.11 |
| baseline_media_movel_4 |    294945 |     5.3826 |     58.2606 |    0.3266 |    0.3676 |   0.588  |     0.6558 |     0.4773 |                100    |
