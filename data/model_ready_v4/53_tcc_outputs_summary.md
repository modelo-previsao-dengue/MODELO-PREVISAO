# Resumo dos Resultados v4

## Resultados Principais (INMET enriquecido, 266 features)

| Variante | R²_log | IC 95% | RMSE | MAE |
|----------|--------|--------|------|-----|
| A (baseline 4yr) | 0.8723 | [0.8661, 0.8780] | 2063.18 | 253.2747 |
| B (série 22yr) | 0.8878 | [0.8820, 0.8929] | 2010.28 | 246.0395 |
| C (série + peso) | 0.8870 | [0.8812, 0.8924] | 1944.84 | 236.9834 |

## Diferenças Estatísticas

| Comparação | ΔR²_log | IC 95% da diferença | p-value | Significância |
|------------|---------|---------------------|---------|---------------|
| B vs A | +0.0155 | [0.0131, 0.0178] | 8.29e-40 | *** |
| C vs A | +0.0147 | [0.0129, 0.0165] | 7.24e-56 | *** |
| C vs B | -0.0008 | [-0.0018, 0.0002] | 1.24e-01 | ns |

## Walk-Forward (INMET enriquecido)

| Variante | R²_log médio |
|----------|-------------|
| A | 0.7934 |
| B | 0.8075 |
| C | 0.8052 |

## Conclusões

1. **A série temporal completa (22 anos) melhora significativamente** a predição de dengue (ΔR²_log = +0.0155, p < 0.001).
2. **O decaimento exponencial (variante C) não melhora** em relação a B (ΔR²_log = -0.0008, ns).
3. **Melhor variante: B** (série completa sem peso).
4. O walk-forward confirma a superioridade de B sobre A (Δ = +0.0141).

## Limitações

1. INMET Silver ausente para 2009, 2020 e 2022 reduz a cobertura climática.
2. A variante A tem cobertura INMET de ~48% (vs ~90% original) devido à falta de 2020/2022.
3. 2026 é ano parcial (17 semanas no momento do corte).
