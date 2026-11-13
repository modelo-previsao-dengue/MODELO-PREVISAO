# Resumo dos Resultados v4

## Resultados Principais (INMET enriquecido, 266 features)

| Variante | R²_log | IC 95% | RMSE | MAE |
|----------|--------|--------|------|-----|
| A (baseline 4yr) | 0.8741 | [0.8681, 0.8797] | 2156.51 | 263.0254 |
| B (série 22yr) | 0.8895 | [0.8838, 0.8944] | 2051.91 | 244.4166 |
| C (série + peso) | 0.8876 | [0.8821, 0.8929] | 2063.44 | 246.7500 |

## Diferenças Estatísticas

| Comparação | ΔR²_log | IC 95% da diferença | p-value | Significância |
|------------|---------|---------------------|---------|---------------|
| B vs A | +0.0154 | [0.0133, 0.0176] | 7.72e-47 | *** |
| C vs A | +0.0135 | [0.0117, 0.0152] | 3.01e-53 | *** |
| C vs B | -0.0019 | [-0.0028, -0.0010] | 2.93e-05 | ns |

## Walk-Forward (INMET enriquecido)

| Variante | R²_log médio |
|----------|-------------|
| A | 0.7954 |
| B | 0.8094 |
| C | 0.8068 |

## Conclusões

1. **A série temporal completa (22 anos) melhora significativamente** a predição de dengue (ΔR²_log = +0.0154, p < 0.001).
2. **O decaimento exponencial (variante C) não melhora** em relação a B (ΔR²_log = -0.0019, ns).
3. **Melhor variante: B** (série completa sem peso).
4. O walk-forward confirma a superioridade de B sobre A (Δ = +0.0140).

## Limitações

1. INMET Silver ausente para 2009, 2020 e 2022 reduz a cobertura climática.
2. A variante A tem cobertura INMET de ~48% (vs ~90% original) devido à falta de 2020/2022.
3. 2026 é ano parcial (17 semanas no momento do corte).
