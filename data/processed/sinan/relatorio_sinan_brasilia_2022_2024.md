# Relatório SINAN - TCC2

## 1. Fontes

- Portal SINAN: https://portalsinan.saude.gov.br/dados-epidemiologicos-SINAN
- OpenDataSUS / S3: https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/json/DENGBR24.json.zip
- Dicionário de dados: https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/dic_dados_dengue.pdf
- Base dos Dados: https://basedosdados.org/dataset/f51134c2-5ab9-4bbc-882f-f1034603147a

## 2. O que o TCC1 realmente fez

- Método: InfoDengue aggregate series
- Recorte: Brasilia/DF
- Período: 2022-2024
- Geocódigo usado: 5300108

## 3. Recorte adotado no TCC2

- Município padrão: Brasilia/DF
- Código de referência: 5300108
- Período: 2022-2024
- Microdados filtrados: 399.094
- Semanas epidemiológicas com atributos: 156

## 4. Coleta por ano

- 2022: 73.245 registros do recorte em 1.405.095 registros totais
- 2023: 43.127 registros do recorte em 1.645.956 registros totais
- 2024: 283.712 registros do recorte em 6.564.924 registros totais

## 5. Clusterização por Intensidade

- Melhor `k`: 2
- Melhor silhouette: 0.1258

Distribuição dos clusters:

- baixa_transmissao: 135 semanas, média 944.1 notificações
- transicao: 21 semanas, média 12935.4 notificações

## 6. Atributos Selecionados por Intensidade

Top atributos com maior capacidade de separar os clusters epidemiológicos:

- NOTIFICACOES: score composto 0.9743
- dor_retro_flag: score composto 0.8740
- febre_flag: score composto 0.6488
- cefaleia_flag: score composto 0.5481
- criterio_clinico_epi: score composto 0.4978
- nausea_flag: score composto 0.4812
- artralgia_flag: score composto 0.4167
- criterio_lab: score composto 0.3906
- caso_dengue_alarme: score composto 0.3846
- laco_flag: score composto 0.3386
- dor_costas_flag: score composto 0.3068
- artrite_flag: score composto 0.2989
- alrm_hepat_flag: score composto 0.2749
- obito_agravo: score composto 0.2703
- grav_melen_flag: score composto 0.2653

## 7. Clusterização por Perfil Clínico-Epidemiológico

- Nesta variante, `NOTIFICACOES` foi excluída do espaço de cluster para reduzir dominância da magnitude semanal.
- Melhor `k`: 6
- Melhor silhouette: 0.1069

Distribuição dos clusters:

- perfil_basal: 24 semanas, média 247.7 notificações
- perfil_intermediario: 24 semanas, média 459.6 notificações
- perfil_alterado: 48 semanas, média 676.8 notificações
- perfil_agudo: 25 semanas, média 2401.6 notificações
- perfil_extremo: 22 semanas, média 2887.2 notificações
- perfil_raro: 13 semanas, média 17390.2 notificações

## 8. Atributos Selecionados por Perfil Clínico

Top atributos discriminativos quando a clusterização ignora o volume bruto de casos:

- raca_parda: score composto 0.9166
- cefaleia_flag: score composto 0.9137
- criterio_lab: score composto 0.8137
- nausea_flag: score composto 0.7697
- dor_retro_flag: score composto 0.7026
- leucopenia_flag: score composto 0.6930
- exantema_flag: score composto 0.6815
- febre_flag: score composto 0.6501
- mialgia_flag: score composto 0.6377
- criterio_clinico_epi: score composto 0.6080
- caso_inconclusivo: score composto 0.5810
- caso_confirmado_provavel: score composto 0.5732
- artrite_flag: score composto 0.5639
- criterio_clinico: score composto 0.5234
- dor_costas_flag: score composto 0.4862

## 9. Observações metodológicas

- O pipeline baixa o arquivo anual completo do SINAN e só depois aplica o recorte analítico.
- A clusterização é feita em nível de semana epidemiológica, para preservar comparabilidade com a série temporal do TCC1.
- A seleção de atributos combina ANOVA, informação mútua e importância de Random Forest contra os rótulos de cluster.
- A análise por intensidade captura semanas epidêmicas versus semanas de baixa transmissão.
- A análise por perfil clínico força a interpretação dos sintomas, sinais de alarme e gravidade sem usar o volume de notificações como variável de agrupamento.