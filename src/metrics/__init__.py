"""Métricas de avaliação para o problema de classificação de sentimento pt-BR.

Implementa a Seção 4.8 do documento mestre
(``projeto-mestrado-analise-sentimentos-ptbr.md``) e espelha
``configs/evaluation.yaml`` -> ``metrics``: métricas de classificação
(F1-macro, MCC), de ranqueamento (ROC-AUC/PR-AUC OvR), de confiança
(calibração, classificação seletiva) e operacionais (latência, custo
computacional), consumidas por ``src/evaluation/evaluator.py``.

Modules
-------
classification
    Métricas de classificação: acurácia, precisão/revocação/F1 (macro e
    por classe), MCC e matriz de confusão.
ranking
    Métricas baseadas em probabilidade no esquema *one-vs-rest*: ROC-AUC e
    PR-AUC.
confidence
    Métricas de confiança das predições: confiança média, correlação
    confiança-acerto, Brier score multiclasse e acurácia sob classificação
    seletiva.
operational
    Métricas de viabilidade operacional: latência de inferência e custo
    computacional (contagem de parâmetros treináveis).
"""

from metrics.classification import (
    calculate_accuracy,
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_matthews_correlation_coefficient,
    calculate_per_class_report,
    calculate_precision_recall_f1,
)
from metrics.confidence import (
    calculate_average_confidence,
    calculate_confidence_accuracy_correlation,
    calculate_multiclass_brier_score,
    calculate_selective_prediction_accuracy,
)
from metrics.operational import (
    calculate_latency_statistics,
    calculate_operational_metrics,
    count_trainable_parameters,
    measure_inference_latency,
)
from metrics.ranking import calculate_pr_auc_ovr, calculate_ranking_metrics, calculate_roc_auc_ovr

__all__: list[str] = [
    "calculate_accuracy",
    "calculate_average_confidence",
    "calculate_classification_metrics",
    "calculate_confidence_accuracy_correlation",
    "calculate_confusion_matrix",
    "calculate_latency_statistics",
    "calculate_matthews_correlation_coefficient",
    "calculate_multiclass_brier_score",
    "calculate_operational_metrics",
    "calculate_per_class_report",
    "calculate_pr_auc_ovr",
    "calculate_precision_recall_f1",
    "calculate_ranking_metrics",
    "calculate_roc_auc_ovr",
    "calculate_selective_prediction_accuracy",
    "count_trainable_parameters",
    "measure_inference_latency",
]
