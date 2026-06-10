# Como subir os dados para o Google Drive

## Passo 1: Subir a pasta de dados

No Google Drive, crie a pasta `TCC2-DADOS` na raiz (My Drive) e suba as subpastas:

```
My Drive/
  TCC2-DADOS/
    sinan/          (copiar de MODELO-PREVISAO/data/sinan/)
    inmet/          (copiar de MODELO-PREVISAO/data/inmet/)
    integrated/     (copiar de MODELO-PREVISAO/data/integrated/)
    model_ready/    (copiar de MODELO-PREVISAO/data/model_ready/)
    reference/      (copiar de MODELO-PREVISAO/data/reference/)
```

Dica: arraste a pasta `data/` inteira para o Drive e renomeie para `TCC2-DADOS`.
Ou use o app Google Drive Desktop para sincronizar automaticamente.

Total: ~2.1 GB (cabe no plano gratis de 15 GB).

## Passo 2: Subir o notebook

Suba o arquivo `TCC2_Dengue_XGBoost.ipynb` para o Google Drive (qualquer lugar).
Depois abra ele com o Google Colab (botao direito > Abrir com > Google Colaboratory).

## Passo 3: Rodar no Colab

O notebook ja faz tudo:
1. Monta o Drive
2. Carrega os dados dos parquets
3. Treina XGBoost regressao e classificacao
4. Compara SINAN-only vs SINAN+INMET
5. Gera graficos SHAP
6. Salva resultados de volta no Drive

## Estrutura final no Drive

```
My Drive/
  TCC2-DADOS/              <- dados (~2.1 GB)
  TCC2-RESULTADOS/         <- criado pelo notebook (metricas + modelos)
  TCC2_Dengue_XGBoost.ipynb  <- notebook Colab
```
