# Model Card (resultados) — Deep Learning

> Gerado automaticamente pela etapa `report`. Complementa o documento escrito à mão [`model_card_deep_learning.md`](./model_card_deep_learning.md) com os números medidos no conjunto de teste.

## Modelos avaliados

`lstm`, `cnn`

## Uso pretendido

Redes LSTM e CNN treinadas do zero sobre o corpus de tweets pt-BR.

## Métricas no conjunto de teste (IC 95% bootstrap)

| Categoria | Modelo | n | F1-macro | MCC | Acurácia | ROC-AUC | Brier | ECE | ms/amostra |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deep Learning | lstm | 12817 | 0.671 [0.663, 0.679] | 0.496 [0.484, 0.508] | 0.669 [0.661, 0.677] | 0.842 | 0.457 | 0.109 | 0.311 |
| Deep Learning | cnn | 12817 | 0.643 [0.635, 0.651] | 0.461 [0.449, 0.474] | 0.641 [0.634, 0.650] | 0.818 | 0.487 | 0.068 | 0.241 |

Melhor modelo da categoria: **lstm** (F1-macro = 0.671, IC 95% [0.663, 0.679]).

## Desempenho por classe

| model | class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- | --- |
| cnn | negativo | 0.65 | 0.632 | 0.641 | 4140.0 |
| cnn | neutro | 0.654 | 0.593 | 0.622 | 5140.0 |
| cnn | positivo | 0.619 | 0.722 | 0.666 | 3537.0 |
| lstm | negativo | 0.648 | 0.647 | 0.647 | 4140.0 |
| lstm | neutro | 0.646 | 0.691 | 0.668 | 5140.0 |
| lstm | positivo | 0.736 | 0.664 | 0.698 | 3537.0 |

## Desempenho por comprimento do texto

| model | slice | n_samples | f1_macro | mcc |
| --- | --- | --- | --- | --- |
| cnn | 1_curto | 4386 | 0.656 | 0.476 |
| cnn | 2_medio | 4216 | 0.637 | 0.449 |
| cnn | 3_longo | 4215 | 0.592 | 0.424 |
| lstm | 1_curto | 4386 | 0.695 | 0.535 |
| lstm | 2_medio | 4216 | 0.666 | 0.487 |
| lstm | 3_longo | 4215 | 0.619 | 0.436 |

## Comparação estatística (McNemar, correção de Holm)

| model_a | model_b | n_common | p_value_holm | significant |
| --- | --- | --- | --- | --- |
| cnn | lstm | 12817 | 0.0 | True |

## Contexto da avaliação

- Tweets de teste: 12817; modelos avaliados no total: 8.
- Melhor modelo geral: `lstm`.

## Limitações e considerações éticas

- Os rótulos de treino e teste vêm de rotuladores automáticos (LLMs); as métricas medem a concordância com esse pseudo-gold, não o acerto sobre a verdade humana. Só um gold set (rotulado por LLM) ou amostra rotulada por humanos mede acerto.
- Intervalos de confiança por bootstrap sobre o conjunto de teste; diferenças entre modelos só devem ser afirmadas quando o McNemar (com correção de Holm) for significativo.
- Nenhum atributo sensível é definido neste domínio; a auditoria de fairness não se aplica.
- Sem monitoramento de drift: o desempenho pode degradar com mudanças de vocabulário.
