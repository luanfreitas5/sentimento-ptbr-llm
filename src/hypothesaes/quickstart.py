"""Fluxo de alto nível do HypotheSAEs: treinar, interpretar, gerar e avaliar hipóteses.

Orquestra os demais módulos de ``hypothesaes`` nas quatro etapas do método
(replicando o notebook ``quickstart.ipynb`` do projeto original):

1. :func:`train_sae` — treina (ou carrega do checkpoint) um Sparse
   Autoencoder sobre embeddings de texto.
2. :func:`interpret_sae` — interpreta uma amostra de neurônios em
   linguagem natural, para checagem de sanidade do SAE treinado.
3. :func:`generate_hypotheses` — seleciona os neurônios mais preditivos de
   uma variável-alvo (ex.: sentimento) e os interpreta como hipóteses.
4. :func:`evaluate_hypotheses` — avalia as hipóteses geradas em um conjunto
   de dados de holdout, medindo o quanto elas realmente predizem o alvo.

É o principal ponto de entrada do subpacote para uso em notebooks e
pipelines (``src/pipelines/``).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hypothesaes.annotate import annotate_texts_with_concepts
from hypothesaes.evaluation import score_hypotheses
from hypothesaes.interpret_neurons import (
    DEFAULT_TASK_SPECIFIC_INSTRUCTIONS,
    InterpretConfig,
    LLMConfig,
    NeuronInterpreter,
    SamplingConfig,
    ScoringConfig,
)
from hypothesaes.sae import SparseAutoencoder, build_sae_checkpoint_name, load_model
from hypothesaes.select_neurons import select_neurons
from hypothesaes.utils import format_text_for_display

logger = logging.getLogger(__name__)


def _infer_classification_task(labels: np.ndarray) -> bool:
    """Infere heuristicamente se o alvo é classificação binária (rótulos em {0, 1})."""
    sample = np.random.default_rng().choice(labels, size=min(1000, len(labels)), replace=True)
    return bool(np.all(np.isin(sample, [0, 1])))


def train_sae(
    embeddings: list | np.ndarray,
    m_total_neurons: int,
    k_active_neurons: int,
    *,
    matryoshka_prefix_lengths: list[int] | None = None,
    batch_topk: bool = False,
    checkpoint_dir: Path | None = None,
    overwrite_checkpoint: bool = False,
    val_embeddings: list | np.ndarray | None = None,
    aux_k: int | None = None,
    multi_k: int | None = None,
    dead_neuron_threshold_steps: int = 256,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    n_epochs: int = 100,
    aux_coef: float = 1 / 32,
    multi_coef: float = 0.0,
    patience: int = 3,
    clip_grad: float = 1.0,
    show_progress: bool = True,
) -> SparseAutoencoder:
    """Treina um Sparse Autoencoder, ou carrega um checkpoint já existente.

    Parameters
    ----------
    embeddings : list | np.ndarray
        Embeddings de treino pré-calculados (ver ``hypothesaes.embedding``).
    m_total_neurons : int
        Número total de neurônios do SAE.
    k_active_neurons : int
        Número de neurônios ativos (top-K) por exemplo.
    matryoshka_prefix_lengths : list[int] | None, optional
        Prefixos para a perda Matryoshka (``None`` para um SAE comum), by
        default None.
    batch_topk : bool, optional
        Se usa esparsidade Top-K em lote, by default False.
    checkpoint_dir : Path | None, optional
        Diretório para salvar/carregar checkpoints do SAE, by default None.
    overwrite_checkpoint : bool, optional
        Se sobrescreve um checkpoint existente em vez de carregá-lo, by
        default False.
    val_embeddings : list | np.ndarray | None, optional
        Embeddings de validação, para early stopping, by default None.
    aux_k : int | None, optional
        Número de neurônios considerados na revivificação de neurônios
        mortos, by default None.
    multi_k : int | None, optional
        Número de neurônios para a reconstrução secundária, by default None.
    dead_neuron_threshold_steps : int, optional
        Passos sem disparo até um neurônio ser considerado morto, by
        default 256.
    batch_size : int, optional
        Tamanho do lote de treino, by default 512.
    learning_rate : float, optional
        Taxa de aprendizado, by default 5e-4.
    n_epochs : int, optional
        Número máximo de épocas de treino, by default 100.
    aux_coef : float, optional
        Coeficiente da perda auxiliar, by default 1/32.
    multi_coef : float, optional
        Coeficiente da perda multi-K, by default 0.0.
    patience : int, optional
        Paciência do early stopping, by default 3.
    clip_grad : float, optional
        Valor de clipping da norma do gradiente, by default 1.0.
    show_progress : bool, optional
        Se exibe barra de progresso do treino, by default True.

    Returns
    -------
    SparseAutoencoder
        Modelo treinado (ou carregado do checkpoint).
    """
    embeddings = np.array(embeddings)
    input_dim = embeddings.shape[1]

    x_train = torch.tensor(embeddings, dtype=torch.float)
    x_val = torch.tensor(val_embeddings, dtype=torch.float) if val_embeddings is not None else None

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / build_sae_checkpoint_name(
            m_total_neurons, k_active_neurons, matryoshka_prefix_lengths
        )
        if checkpoint_path.is_file() and not overwrite_checkpoint:
            return load_model(checkpoint_path)

    sae = SparseAutoencoder(
        input_dim=input_dim,
        m_total_neurons=m_total_neurons,
        k_active_neurons=k_active_neurons,
        aux_k=aux_k,
        multi_k=multi_k,
        dead_neuron_threshold_steps=dead_neuron_threshold_steps,
        prefix_lengths=matryoshka_prefix_lengths,
        use_batch_topk=batch_topk,
    )

    sae.fit(
        x_train=x_train,
        x_val=x_val,
        save_dir=checkpoint_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        aux_coef=aux_coef,
        multi_coef=multi_coef,
        patience=patience,
        clip_grad=clip_grad,
        show_progress=show_progress,
    )

    return sae


def _resolve_neuron_indices(
    activation_counts: np.ndarray,
    total_neurons: int,
    neuron_indices: list[int] | None,
    n_random_neurons: int | None,
    n_top_neurons: int | None,
) -> list[int]:
    """Resolve os índices de neurônios a interpretar, a partir da seleção informada."""
    if neuron_indices is not None:
        return neuron_indices

    if n_random_neurons is not None:
        random_indices = np.random.default_rng().choice(
            total_neurons, size=n_random_neurons, replace=False
        )
        return [int(i) for i in random_indices]

    assert n_top_neurons is not None, (
        "n_top_neurons não pode ser None quando neuron_indices e n_random_neurons também são None"
    )
    if n_top_neurons > total_neurons:
        raise ValueError(
            f"n_top_neurons ({n_top_neurons}) não pode exceder o total de "
            f"neurônios ({total_neurons})"
        )
    top_indices = np.argsort(activation_counts)[-n_top_neurons:][::-1]
    return [int(i) for i in top_indices]


def _build_neuron_interpretation_result(
    idx: int,
    activations: np.ndarray,
    activation_percent: np.ndarray,
    interpretations: dict[int, list[str | None]],
    n_candidates: int,
    texts: list[str],
    print_examples_n: int,
    print_examples_max_chars: int,
) -> dict[str, Any]:
    """Monta a linha de resultado de um neurônio interpretado,
    com exemplos de ativação opcionais.
    """
    neuron_activations = activations[:, idx]
    result: dict[str, Any] = {
        "neuron_idx": int(idx),
        "interpretation": interpretations[idx][0] if n_candidates == 1 else interpretations[idx],
    }

    if print_examples_n <= 0:
        return result

    top_indices = np.argsort(neuron_activations)[-print_examples_n:][::-1]
    top_examples = [texts[i] for i in top_indices]
    logger.info(
        "Neurônio %d (%.1f%% ativo): %s", idx, activation_percent[idx], interpretations[idx][0]
    )
    for i, example in enumerate(top_examples, 1):
        logger.info(
            "%d. %s", i, format_text_for_display(example, max_chars=print_examples_max_chars)
        )
        result[f"top_example_{i}"] = example

    return result


def interpret_sae(
    texts: list[str],
    embeddings: list | np.ndarray,
    sae: SparseAutoencoder,
    *,
    neuron_indices: list[int] | None = None,
    n_random_neurons: int | None = None,
    n_top_neurons: int | None = None,
    interpreter_model: str = "gpt-5.2",
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpret_temperature: float = 0.7,
    max_interpretation_tokens: int | None = None,
    interpret_llm_kwargs: dict[str, Any] | None = None,
    n_candidates: int = 1,
    print_examples_n: int = 3,
    print_examples_max_chars: int = 1024,
    task_specific_instructions: str | None = None,
) -> pd.DataFrame:
    """Interpreta uma amostra de neurônios de um SAE treinado, para checagem de sanidade.

    Parameters
    ----------
    texts : list[str]
        Textos de entrada correspondentes aos ``embeddings``.
    embeddings : list | np.ndarray
        Embeddings pré-calculados dos textos de entrada.
    sae : SparseAutoencoder
        Um SAE já treinado.
    neuron_indices : list[int] | None, optional
        Índices específicos de neurônios a interpretar (mutuamente
        exclusivo com ``n_random_neurons``/``n_top_neurons``), by default
        None.
    n_random_neurons : list[int] | None, optional
        Número de neurônios aleatórios a interpretar, by default None.
    n_top_neurons : int | None, optional
        Número dos neurônios mais prevalentes a interpretar, by default
        None.
    interpreter_model : str, optional
        LLM usado para gerar interpretações, by default "gpt-5.2".
    n_examples_for_interpretation : int, optional
        Número de exemplos usados no prompt de interpretação, by default 20.
    max_words_per_example : int, optional
        Máximo de palavras por exemplo enviado ao LLM intérprete, by
        default 256.
    interpret_temperature : float, optional
        Temperatura do LLM intérprete, by default 0.7.
    max_interpretation_tokens : int | None, optional
        Máximo de tokens da interpretação gerada; ``None`` não limita, by
        default None.
    interpret_llm_kwargs : dict[str, Any] | None, optional
        Argumentos extras repassados à API do LLM, by default None.
    n_candidates : int, optional
        Número de interpretações candidatas por neurônio, by default 1.
    print_examples_n : int, optional
        Número de exemplos de maior ativação a exibir (0 desabilita), by
        default 3.
    print_examples_max_chars : int, optional
        Máximo de caracteres exibidos por exemplo, by default 1024.
    task_specific_instructions : str | None, optional
        Instruções específicas da tarefa, incluídas no prompt de
        interpretação, by default None.

    Returns
    -------
    pd.DataFrame
        Uma linha por neurônio interpretado, com ``neuron_idx``,
        ``interpretation`` e, se ``print_examples_n > 0``, colunas
        ``top_example_{i}``.

    Raises
    ------
    ValueError
        Se não for informado exatamente um entre ``neuron_indices``,
        ``n_random_neurons`` e ``n_top_neurons``.
    """
    selection_params = [neuron_indices, n_random_neurons, n_top_neurons]
    if sum(param is not None for param in selection_params) != 1:
        raise ValueError(
            "É necessário informar exatamente um entre neuron_indices, "
            "n_random_neurons e n_top_neurons"
        )

    x = (
        embeddings
        if isinstance(embeddings, torch.Tensor)
        else torch.tensor(embeddings, dtype=torch.float)
    )

    activations = sae.compute_activations(x)
    logger.info("Formato das ativações: %s", activations.shape)
    activation_counts = (activations != 0).sum(axis=0)
    activation_percent = activation_counts / activations.shape[0] * 100

    total_neurons = activations.shape[1]
    neuron_indices = _resolve_neuron_indices(
        activation_counts, total_neurons, neuron_indices, n_random_neurons, n_top_neurons
    )

    interpreter = NeuronInterpreter(interpreter_model=interpreter_model)

    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            n_examples=n_examples_for_interpretation, max_words_per_example=max_words_per_example
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_output_tokens=max_interpretation_tokens,
            llm_kwargs=interpret_llm_kwargs or {},
        ),
        n_candidates=n_candidates,
        task_specific_instructions=task_specific_instructions or DEFAULT_TASK_SPECIFIC_INSTRUCTIONS,
    )

    interpretations = interpreter.interpret_neurons(
        texts=texts, activations=activations, neuron_indices=neuron_indices, config=interpret_config
    )

    results_list = [
        _build_neuron_interpretation_result(
            idx,
            activations,
            activation_percent,
            interpretations,
            n_candidates,
            texts,
            print_examples_n,
            print_examples_max_chars,
        )
        for idx in neuron_indices
    ]

    return pd.DataFrame(results_list)


def _build_hypothesis_rows_without_scoring(
    selected_neurons: list[int],
    scores: list[float],
    interpretations: dict[int, list[str | None]],
    selection_method: str,
) -> list[dict[str, Any]]:
    """Monta as linhas de resultado sem pontuar a fidelidade das interpretações."""
    return [
        {
            "neuron_idx": idx,
            f"target_{selection_method}": score,
            "interpretation": interpretations[idx][0],
        }
        for idx, score in zip(selected_neurons, scores, strict=True)
    ]


def _build_hypothesis_rows_with_scoring(
    selected_neurons: list[int],
    scores: list[float],
    interpretations: dict[int, list[str | None]],
    metrics: dict[int, dict[str | None, dict[str, float]]],
    selection_method: str,
    scoring_metric: str,
) -> list[dict[str, Any]]:
    """Monta as linhas de resultado, escolhendo a interpretação de maior fidelidade por neurônio."""
    rows = []
    for idx, score in zip(selected_neurons, scores, strict=True):
        best_interpretation = max(
            interpretations[idx], key=lambda interp: metrics[idx][interp][scoring_metric]
        )
        rows.append(
            {
                "neuron_idx": idx,
                f"target_{selection_method}": score,
                "interpretation": best_interpretation,
                f"{scoring_metric}_fidelity_score": metrics[idx][best_interpretation][
                    scoring_metric
                ],
            }
        )
    return rows


def generate_hypotheses(
    texts: list[str],
    labels: list[int] | list[float] | np.ndarray,
    embeddings: list | np.ndarray,
    sae: SparseAutoencoder,
    *,
    cache_name: str | None = None,
    classification: bool | None = None,
    selection_method: str = "separation_score",
    n_selected_neurons: int = 20,
    interpreter_model: str = "gpt-5.2",
    annotator_model: str = "gpt-5-mini",
    n_examples_for_interpretation: int = 20,
    max_words_per_example: int = 256,
    interpret_temperature: float = 0.7,
    max_interpretation_tokens: int | None = None,
    interpret_llm_kwargs: dict[str, Any] | None = None,
    annotation_llm_kwargs: dict[str, Any] | None = None,
    n_candidate_interpretations: int = 1,
    n_scoring_examples: int = 100,
    scoring_metric: str = "f1",
    n_workers_interpretation: int = 10,
    n_workers_annotation: int = 30,
    task_specific_instructions: str | None = None,
) -> pd.DataFrame:
    """Gera hipóteses interpretáveis a partir de texto, usando um SAE treinado.

    Etapas: (1) seleciona os neurônios mais preditivos do alvo; (2)
    interpreta cada neurônio selecionado em linguagem natural; (3) pontua
    a fidelidade de cada interpretação (opcional, ver ``n_scoring_examples``).

    Parameters
    ----------
    texts : list[str]
        Textos de entrada.
    labels : list[int] | list[float] | np.ndarray
        Rótulos-alvo (binários para classificação, contínuos para
        regressão).
    embeddings : list | np.ndarray
        Embeddings pré-calculados dos textos de entrada.
    sae : SparseAutoencoder
        Um SAE já treinado.
    cache_name : str | None, optional
        Prefixo do cache de anotações, by default None.
    classification : bool | None, optional
        Se é uma tarefa de classificação; se ``None``, é inferido a
        partir de ``labels``, by default None.
    selection_method : str, optional
        Método de seleção de neurônios preditivos ('separation_score',
        'correlation', 'lasso'), by default "separation_score".
    n_selected_neurons : int, optional
        Número de neurônios a selecionar e interpretar, by default 20.
    interpreter_model : str, optional
        LLM usado para gerar interpretações, by default "gpt-5.2".
    annotator_model : str, optional
        LLM usado para pontuar interpretações, by default "gpt-5-mini".
    n_examples_for_interpretation : int, optional
        Número de exemplos usados no prompt de interpretação, by default 20.
    max_words_per_example : int, optional
        Máximo de palavras por exemplo enviado ao LLM, by default 256.
    interpret_temperature : float, optional
        Temperatura do LLM intérprete, by default 0.7.
    max_interpretation_tokens : int | None, optional
        Máximo de tokens da interpretação gerada, by default None.
    interpret_llm_kwargs : dict[str, Any] | None, optional
        Argumentos extras para as requisições de interpretação, by
        default None.
    annotation_llm_kwargs : dict[str, Any] | None, optional
        Argumentos extras para as requisições de anotação/pontuação, by
        default None.
    n_candidate_interpretations : int, optional
        Número de interpretações candidatas por neurônio, by default 1.
    n_scoring_examples : int, optional
        Número de exemplos usados para pontuar interpretações (0 desabilita
        a pontuação), by default 100.
    scoring_metric : str, optional
        Métrica usada para ranquear interpretações candidatas ('f1',
        'precision', 'recall', 'correlation'), by default "f1".
    n_workers_interpretation : int, optional
        Threads paralelas para interpretação, by default 10.
    n_workers_annotation : int, optional
        Threads paralelas para anotação, by default 30.
    task_specific_instructions : str | None, optional
        Instruções específicas da tarefa, incluídas no prompt de
        interpretação, by default None.

    Returns
    -------
    pd.DataFrame
        Colunas: ``neuron_idx``, ``target_{selection_method}``,
        ``interpretation`` e, se pontuado, ``{scoring_metric}_fidelity_score``.

    Raises
    ------
    ValueError
        Se ``n_selected_neurons`` exceder o total de neurônios do SAE.
    """
    labels = np.array(labels)
    x = (
        embeddings
        if isinstance(embeddings, torch.Tensor)
        else torch.tensor(embeddings, dtype=torch.float)
    )

    if classification is None:
        classification = _infer_classification_task(labels)

    logger.info("Formato dos embeddings: %s", np.shape(embeddings))

    activations = sae.compute_activations(x)
    logger.info("Formato das ativações: %s", activations.shape)

    logger.info("Etapa 1: selecionando os %d neurônios mais preditivos", n_selected_neurons)
    if n_selected_neurons > activations.shape[1]:
        raise ValueError(
            f"n_selected_neurons ({n_selected_neurons}) pode ser no máximo o total de "
            f"neurônios ({activations.shape[1]})"
        )

    selected_neurons, scores = select_neurons(
        activations=activations,
        target=labels,
        n_select=n_selected_neurons,
        method=selection_method,
        classification=classification,
        verbose=True,
    )

    logger.info("Etapa 2: interpretando os neurônios selecionados")
    interpreter = NeuronInterpreter(
        cache_name=cache_name,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model,
        n_workers_interpretation=n_workers_interpretation,
        n_workers_annotation=n_workers_annotation,
    )

    interpret_config = InterpretConfig(
        sampling=SamplingConfig(
            n_examples=n_examples_for_interpretation, max_words_per_example=max_words_per_example
        ),
        llm=LLMConfig(
            temperature=interpret_temperature,
            max_output_tokens=max_interpretation_tokens,
            llm_kwargs=interpret_llm_kwargs or {},
        ),
        n_candidates=n_candidate_interpretations,
        task_specific_instructions=task_specific_instructions or DEFAULT_TASK_SPECIFIC_INSTRUCTIONS,
    )

    interpretations = interpreter.interpret_neurons(
        texts=texts,
        activations=activations,
        neuron_indices=selected_neurons,
        config=interpret_config,
    )

    if n_scoring_examples == 0:
        results = _build_hypothesis_rows_without_scoring(
            selected_neurons, scores, interpretations, selection_method
        )
        return pd.DataFrame(results)

    logger.info("Etapa 3: pontuando interpretações")
    scoring_config = ScoringConfig(n_examples=n_scoring_examples)
    metrics = interpreter.score_interpretations(
        texts=texts,
        activations=activations,
        interpretations=interpretations,
        config=scoring_config,
        **(annotation_llm_kwargs or {}),
    )

    results = _build_hypothesis_rows_with_scoring(
        selected_neurons, scores, interpretations, metrics, selection_method, scoring_metric
    )
    return pd.DataFrame(results)


def evaluate_hypotheses(
    hypotheses_df: pd.DataFrame,
    texts: list[str],
    labels: list[int] | list[float] | np.ndarray,
    *,
    cache_name: str | None = None,
    annotator_model: str = "gpt-5-mini",
    max_words_per_example: int = 256,
    classification: bool | None = None,
    n_workers_annotation: int = 30,
    corrected_pval_threshold: float = 0.1,
    annotation_llm_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, float | tuple[int, int, float]], pd.DataFrame]:
    """Avalia hipóteses geradas em um conjunto de dados de holdout.

    Parameters
    ----------
    hypotheses_df : pd.DataFrame
        DataFrame retornado por :func:`generate_hypotheses`.
    texts : list[str]
        Textos de holdout.
    labels : list[int] | list[float] | np.ndarray
        Rótulos de holdout.
    cache_name : str | None, optional
        Prefixo do cache de anotações, by default None.
    annotator_model : str, optional
        Modelo usado na anotação, by default "gpt-5-mini".
    max_words_per_example : int, optional
        Máximo de palavras por exemplo anotado, by default 256.
    classification : bool | None, optional
        Se é uma tarefa de classificação; se ``None``, é inferido a
        partir de ``labels``, by default None.
    n_workers_annotation : int, optional
        Threads paralelas para anotação, by default 30.
    corrected_pval_threshold : float, optional
        Limiar de significância antes da correção de Bonferroni, by
        default 0.1.
    annotation_llm_kwargs : dict[str, Any] | None, optional
        Argumentos extras para as requisições de anotação, by default None.

    Returns
    -------
    tuple[dict[str, float | tuple[int, int, float]], pd.DataFrame]
        Par (métricas agregadas, DataFrame de avaliação por hipótese),
        conforme :func:`hypothesaes.evaluation.score_hypotheses`.
    """
    labels = np.array(labels)

    if classification is None:
        classification = _infer_classification_task(labels)

    hypotheses = hypotheses_df["interpretation"].tolist()

    logger.info("Etapa 1: anotando textos com %d hipótese(s)", len(hypotheses))
    hypothesis_annotations = annotate_texts_with_concepts(
        texts=texts,
        concepts=hypotheses,
        max_words_per_example=max_words_per_example,
        model=annotator_model,
        cache_name=cache_name,
        n_workers=n_workers_annotation,
        **(annotation_llm_kwargs or {}),
    )

    logger.info("Etapa 2: calculando o poder preditivo das anotações")
    metrics, evaluation_df = score_hypotheses(
        hypothesis_annotations=hypothesis_annotations,
        y_true=labels,
        classification=classification,
        corrected_pval_threshold=corrected_pval_threshold,
    )

    return metrics, evaluation_df
