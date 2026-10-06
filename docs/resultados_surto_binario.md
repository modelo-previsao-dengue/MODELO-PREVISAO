# Resultados do surto binario (US-010)

Alvo: surto/nao-surto por canal endemico (mesorregiao, teste 2023).
`via_regressao` limiariza a previsao continua da regressao pelo canal, sem parametro livre. `classificador` e um XGBClassifier binario dedicado, limiar calibrado na metade rigorosa da validacao (SE>=27, nunca vista por Optuna nem early stopping).

## Variante do canal: `media`

| modelo                                  |   n_teste |   prevalencia |   limiar |   precisao |   recall |   especificidade |   F1_positivo |   F2_positivo |   AUC_ROC |   AUC_PR |   VP |   FP |   VN |   FN |
|:----------------------------------------|----------:|--------------:|---------:|-----------:|---------:|-----------------:|--------------:|--------------:|----------:|---------:|-----:|-----:|-----:|-----:|
| baseline: persistencia_de_rotulo        |      2279 |        0.5994 | nan      |     0.8724 |   0.8206 |           0.8204 |        0.8457 |        0.8305 |  nan      | nan      | 1121 |  164 |  749 |  245 |
| meso_surto_sinan :: via_regressao       |      2279 |        0.5994 | nan      |     0.8577 |   0.8075 |           0.7996 |        0.8318 |        0.817  |    0.6644 |   0.7426 | 1103 |  183 |  730 |  263 |
| meso_surto_sinan_inmet :: via_regressao |      2279 |        0.5994 | nan      |     0.8627 |   0.8001 |           0.8094 |        0.8302 |        0.8119 |    0.6588 |   0.7356 | 1093 |  174 |  739 |  273 |
| baseline: persistencia                  |      2279 |        0.5994 | nan      |     0.8434 |   0.8082 |           0.7755 |        0.8254 |        0.815  |    0.6701 |   0.7574 | 1104 |  205 |  708 |  262 |
| baseline: media_movel_4                 |      2279 |        0.5994 | nan      |     0.8208 |   0.798  |           0.7393 |        0.8092 |        0.8024 |    0.6859 |   0.7658 | 1090 |  238 |  675 |  276 |
| baseline: sempre_positivo               |      2279 |        0.5994 | nan      |     0.5994 |   1      |           0      |        0.7495 |        0.8821 |  nan      | nan      | 1366 |  913 |    0 |    0 |
| meso_surto_sinan_inmet :: classificador |      2279 |        0.5994 |   0.2626 |     0.6848 |   0.8031 |           0.4469 |        0.7392 |        0.7763 |    0.6757 |   0.7392 | 1097 |  505 |  408 |  269 |
| meso_surto_sinan :: classificador       |      2279 |        0.5994 |   0.2875 |     0.6471 |   0.8163 |           0.3341 |        0.7219 |        0.7757 |    0.6426 |   0.7171 | 1115 |  608 |  305 |  251 |
| baseline: sazonal                       |      2236 |        0.5948 | nan      |     0.6761 |   0.4301 |           0.6976 |        0.5257 |        0.4638 |    0.5363 |   0.5981 |  572 |  274 |  632 |  758 |
| baseline: sempre_negativo               |      2279 |        0.5994 | nan      |     0      |   0      |           1      |        0      |        0      |  nan      | nan      |    0 |    0 |  913 | 1366 |

## Variante do canal: `media_2dp`

| modelo                                  |   n_teste |   prevalencia |   limiar |   precisao |   recall |   especificidade |   F1_positivo |   F2_positivo |   AUC_ROC |   AUC_PR |   VP |   FP |   VN |   FN |
|:----------------------------------------|----------:|--------------:|---------:|-----------:|---------:|-----------------:|--------------:|--------------:|----------:|---------:|-----:|-----:|-----:|-----:|
| baseline: persistencia_de_rotulo        |      2279 |        0.2799 | nan      |     0.7437 |   0.6912 |           0.9074 |        0.7165 |        0.7011 |  nan      | nan      |  441 |  152 | 1489 |  197 |
| baseline: persistencia                  |      2279 |        0.2799 | nan      |     0.622  |   0.6473 |           0.847  |        0.6344 |        0.6421 |    0.6415 |   0.4512 |  413 |  251 | 1390 |  225 |
| meso_surto_sinan :: via_regressao       |      2279 |        0.2799 | nan      |     0.6927 |   0.5831 |           0.8995 |        0.6332 |        0.6021 |    0.6428 |   0.4272 |  372 |  165 | 1476 |  266 |
| meso_surto_sinan_inmet :: via_regressao |      2279 |        0.2799 | nan      |     0.7175 |   0.5533 |           0.9153 |        0.6248 |        0.5798 |    0.6385 |   0.4143 |  353 |  139 | 1502 |  285 |
| baseline: media_movel_4                 |      2279 |        0.2799 | nan      |     0.5449 |   0.5987 |           0.8056 |        0.5706 |        0.5872 |    0.6451 |   0.4502 |  382 |  319 | 1322 |  256 |
| baseline: sempre_positivo               |      2279 |        0.2799 | nan      |     0.2799 |   1      |           0      |        0.4374 |        0.6603 |  nan      | nan      |  638 | 1641 |    0 |    0 |
| meso_surto_sinan :: classificador       |      2279 |        0.2799 |   0.2198 |     0.3313 |   0.3433 |           0.7307 |        0.3372 |        0.3408 |    0.5819 |   0.3845 |  219 |  442 | 1199 |  419 |
| meso_surto_sinan_inmet :: classificador |      2279 |        0.2799 |   0.3404 |     0.468  |   0.2179 |           0.9037 |        0.2973 |        0.2439 |    0.6373 |   0.4219 |  139 |  158 | 1483 |  499 |
| baseline: sazonal                       |      2236 |        0.2773 | nan      |     0.4377 |   0.221  |           0.8911 |        0.2937 |        0.2453 |    0.5415 |   0.2822 |  137 |  176 | 1440 |  483 |
| baseline: sempre_negativo               |      2279 |        0.2799 | nan      |     0      |   0      |           1      |        0      |        0      |  nan      | nan      |    0 |    0 | 1641 |  638 |

