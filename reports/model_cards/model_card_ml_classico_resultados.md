# Model Card (resultados) — Baseline + ML tradicional

> Gerado automaticamente pela etapa `report`. Complementa o documento escrito à mão [`model_card_ml_classico.md`](./model_card_ml_classico.md) com os números medidos no conjunto de teste.

## Modelos avaliados

`naive_bayes`, `logistic_regression`, `svm`, `gradient_boosting`, `random_forest`, `dummy`

## Uso pretendido

Baseline interpretável e de baixo custo (dummy, Naive Bayes, Regressão Logística, SVM, Random Forest, Gradient Boosting sobre TF-IDF) para classificação de sentimento em tweets pt-BR.

## Métricas no conjunto de teste (IC 95% bootstrap)

| Categoria | Modelo | n | F1-macro | MCC | Acurácia | ROC-AUC | Brier | ECE | ms/amostra |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline + ML tradicional | naive_bayes | 12817 | 0.636 [0.627, 0.644] | 0.450 [0.437, 0.463] | 0.634 [0.626, 0.643] | 0.797 | 0.690 | 0.334 | 0.051 |
| Baseline + ML tradicional | logistic_regression | 12817 | 0.622 [0.613, 0.630] | 0.424 [0.412, 0.438] | 0.621 [0.613, 0.630] | 0.776 | 0.651 | 0.279 | 0.009 |
| Baseline + ML tradicional | svm | 12817 | 0.607 [0.598, 0.616] | 0.420 [0.408, 0.433] | 0.618 [0.609, 0.626] | 0.786 | 0.528 | 0.108 | 0.153 |
| Baseline + ML tradicional | gradient_boosting | 12817 | 0.589 [0.581, 0.598] | 0.398 [0.386, 0.411] | 0.604 [0.595, 0.612] | 0.782 | 0.521 | 0.066 | 0.013 |
| Baseline + ML tradicional | random_forest | 12817 | 0.575 [0.566, 0.583] | 0.355 [0.342, 0.368] | 0.580 [0.571, 0.588] | 0.751 | 0.618 | 0.196 | 0.021 |
| Baseline + ML tradicional | dummy | 12817 | 0.336 [0.328, 0.344] | 0.004 [-0.008, 0.017] | 0.344 [0.336, 0.352] | 0.502 | 1.312 | 0.656 | 0.000 |

Melhor modelo da categoria: **naive_bayes** (F1-macro = 0.636, IC 95% [0.627, 0.644]).

## Desempenho por classe

| model | class | precision | recall | f1 | support |
| --- | --- | --- | --- | --- | --- |
| dummy | negativo | 0.327 | 0.326 | 0.327 | 4140.0 |
| dummy | neutro | 0.404 | 0.401 | 0.402 | 5140.0 |
| dummy | positivo | 0.278 | 0.281 | 0.279 | 3537.0 |
| gradient_boosting | negativo | 0.638 | 0.484 | 0.55 | 4140.0 |
| gradient_boosting | neutro | 0.542 | 0.808 | 0.649 | 5140.0 |
| gradient_boosting | positivo | 0.787 | 0.446 | 0.569 | 3537.0 |
| logistic_regression | negativo | 0.627 | 0.608 | 0.618 | 4140.0 |
| logistic_regression | neutro | 0.611 | 0.628 | 0.62 | 5140.0 |
| logistic_regression | positivo | 0.629 | 0.627 | 0.628 | 3537.0 |
| naive_bayes | negativo | 0.618 | 0.684 | 0.649 | 4140.0 |
| naive_bayes | neutro | 0.654 | 0.573 | 0.611 | 5140.0 |
| naive_bayes | positivo | 0.63 | 0.665 | 0.647 | 3537.0 |
| random_forest | negativo | 0.557 | 0.543 | 0.55 | 4140.0 |
| random_forest | neutro | 0.548 | 0.68 | 0.607 | 5140.0 |
| random_forest | positivo | 0.7 | 0.476 | 0.567 | 3537.0 |
| svm | negativo | 0.679 | 0.499 | 0.575 | 4140.0 |
| svm | neutro | 0.554 | 0.801 | 0.655 | 5140.0 |
| svm | positivo | 0.742 | 0.491 | 0.591 | 3537.0 |

## Desempenho por comprimento do texto

| model | slice | n_samples | f1_macro | mcc |
| --- | --- | --- | --- | --- |
| dummy | 1_curto | 4386 | 0.332 | 0.002 |
| dummy | 2_medio | 4216 | 0.332 | -0.0 |
| dummy | 3_longo | 4215 | 0.337 | 0.012 |
| gradient_boosting | 1_curto | 4386 | 0.526 | 0.393 |
| gradient_boosting | 2_medio | 4216 | 0.567 | 0.372 |
| gradient_boosting | 3_longo | 4215 | 0.598 | 0.398 |
| logistic_regression | 1_curto | 4386 | 0.643 | 0.458 |
| logistic_regression | 2_medio | 4216 | 0.595 | 0.384 |
| logistic_regression | 3_longo | 4215 | 0.604 | 0.407 |
| naive_bayes | 1_curto | 4386 | 0.634 | 0.442 |
| naive_bayes | 2_medio | 4216 | 0.631 | 0.445 |
| naive_bayes | 3_longo | 4215 | 0.621 | 0.44 |
| random_forest | 1_curto | 4386 | 0.5 | 0.346 |
| random_forest | 2_medio | 4216 | 0.573 | 0.348 |
| random_forest | 3_longo | 4215 | 0.531 | 0.341 |
| svm | 1_curto | 4386 | 0.599 | 0.439 |
| svm | 2_medio | 4216 | 0.587 | 0.384 |
| svm | 3_longo | 4215 | 0.604 | 0.409 |

## Comparação estatística (McNemar, correção de Holm)

| model_a | model_b | n_common | p_value_holm | significant |
| --- | --- | --- | --- | --- |
| dummy | gradient_boosting | 12817 | 0.0 | True |
| dummy | logistic_regression | 12817 | 0.0 | True |
| dummy | naive_bayes | 12817 | 0.0 | True |
| dummy | random_forest | 12817 | 0.0 | True |
| dummy | svm | 12817 | 0.0 | True |
| gradient_boosting | logistic_regression | 12817 | 0.0036 | True |
| gradient_boosting | naive_bayes | 12817 | 0.0 | True |
| gradient_boosting | random_forest | 12817 | 0.0 | True |
| gradient_boosting | svm | 12817 | 0.0055 | True |
| logistic_regression | naive_bayes | 12817 | 0.0122 | True |
| logistic_regression | random_forest | 12817 | 0.0 | True |
| logistic_regression | svm | 12817 | 0.4187 | False |
| naive_bayes | random_forest | 12817 | 0.0 | True |
| naive_bayes | svm | 12817 | 0.0051 | True |
| random_forest | svm | 12817 | 0.0 | True |

## Ablação do pipeline (conjunto de validação)

| component | baseline_value | ablated_value | impact | impact_ci_low | impact_ci_high |
| --- | --- | --- | --- | --- | --- |
| sem_tf_sublinear | 0.6303 | 0.6301 | 0.0002 | -0.0033 | 0.0037 |
| sem_filtro_frequencia_minima | 0.6303 | 0.6303 | 0.0 | 0.0 | 0.0 |
| sem_class_weight | 0.6303 | 0.6314 | -0.0011 | -0.004 | 0.002 |
| sem_bigramas | 0.6303 | 0.6373 | -0.0069 | -0.0189 | 0.0051 |
| sem_limite_vocabulario | 0.6303 | 0.653 | -0.0227 | -0.0339 | -0.0126 |

## Contexto da avaliação

- Tweets de teste: 12817; modelos avaliados no total: 8.
- Melhor modelo geral: `lstm`.

## Limitações e considerações éticas

- Os rótulos de treino e teste vêm de rotuladores automáticos (LLMs); as métricas medem a concordância com esse pseudo-gold, não o acerto sobre a verdade humana. Só um gold set (rotulado por LLM) ou amostra rotulada por humanos mede acerto.
- Intervalos de confiança por bootstrap sobre o conjunto de teste; diferenças entre modelos só devem ser afirmadas quando o McNemar (com correção de Holm) for significativo.
- Nenhum atributo sensível é definido neste domínio; a auditoria de fairness não se aplica.
- Sem monitoramento de drift: o desempenho pode degradar com mudanças de vocabulário.
