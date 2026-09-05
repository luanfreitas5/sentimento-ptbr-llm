"""Avaliação rigorosa de classificadores de sentimento pt-BR.

Implementa a Seção 4.9 do documento mestre
(``projeto-mestrado-analise-sentimentos-ptbr.md``) e a diretriz "Rigorous
evaluation" do CLAUDE.md: métricas pontuais sempre acompanhadas de
incerteza (bootstrap), comparação entre modelos com testes de
significância estatística, calibração de probabilidades, avaliação por
fatia dos dados e *ablation study* dos componentes do pipeline.

Modules
-------
evaluator
    Orquestra as métricas de ``src/metrics/`` em um :class:`EvaluationResult`
    com intervalos de confiança bootstrap.
significance
    Testes de McNemar, Wilcoxon, Friedman e post-hoc de Nemenyi para
    comparação entre classificadores (``configs/evaluation.yaml`` ->
    ``significance_tests``).
calibration
    Curva de confiabilidade, Erro de Calibração Esperado (ECE) e Brier
    score multiclasse.
slice_evaluation
    Métricas de classificação por fatia dos dados (fonte, classe,
    comprimento de texto) e identificação de fatias com desempenho abaixo
    do limiar.
ablation
    Impacto da remoção de cada componente do pipeline sobre a métrica
    principal.
reports
    Consolidação e persistência de relatórios de avaliação comparáveis
    entre modelos.
"""

from evaluation.ablation import calculate_ablation_impact, identify_most_impactful_component
from evaluation.calibration import (
    calculate_calibration_metrics,
    calculate_expected_calibration_error,
    calculate_reliability_curve,
)
from evaluation.evaluator import (
    EvaluationResult,
    calculate_bootstrap_confidence_intervals,
    evaluate_classifier,
)
from evaluation.reports import (
    build_evaluation_report,
    merge_evaluation_reports,
    save_evaluation_report,
)
from evaluation.significance import (
    run_friedman_test,
    run_mcnemar_test,
    run_nemenyi_post_hoc_test,
    run_wilcoxon_signed_rank_test,
)
from evaluation.slice_evaluation import evaluate_metrics_by_slice, identify_underperforming_slices

__all__: list[str] = [
    "EvaluationResult",
    "build_evaluation_report",
    "calculate_ablation_impact",
    "calculate_bootstrap_confidence_intervals",
    "calculate_calibration_metrics",
    "calculate_expected_calibration_error",
    "calculate_reliability_curve",
    "evaluate_classifier",
    "evaluate_metrics_by_slice",
    "identify_most_impactful_component",
    "identify_underperforming_slices",
    "merge_evaluation_reports",
    "run_friedman_test",
    "run_mcnemar_test",
    "run_nemenyi_post_hoc_test",
    "run_wilcoxon_signed_rank_test",
    "save_evaluation_report",
]
