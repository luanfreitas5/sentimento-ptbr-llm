"""Interpretação de neurônios do SAE em linguagem natural, via LLM.

Converte um neurônio do Sparse Autoencoder (identificado por seus tweets de
ativação mais alta e por tweets sem ativação) em uma hipótese textual curta
(ex.: "reclama do tempo de espera no atendimento"), usando um LLM como
intérprete (:class:`NeuronInterpreter`). Também mede a *fidelidade* de cada
interpretação: o quanto ela, quando usada como critério de anotação
(``annotate.py``), reproduz o padrão real de ativação do neurônio.
"""

import concurrent.futures
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from hypothesaes.annotate import ANNOTATION_CACHE_DIR, annotate_tasks
from hypothesaes.llm_api import generate_completion, normalize_llm_kwargs
from hypothesaes.utils import load_prompt_template, truncate_text

logger = logging.getLogger(__name__)

DEFAULT_TASK_SPECIFIC_INSTRUCTIONS = """An example feature could be:
- "uses multiple adjectives to describe colors"
- "describes a patient experiencing seizures or epilepsy"
- "contains multiple single-digit numbers\""""


def sample_top_zero(
    texts: list[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: int | None = None,
    random_seed: int | None = None,
) -> dict[str, list[str] | list[float]]:
    """Amostra os exemplos de maior ativação e exemplos aleatórios de ativação zero.

    Parameters
    ----------
    texts : list[str]
        Todos os textos do corpus.
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    neuron_idx : int
        Índice do neurônio a amostrar.
    n_examples : int
        Número total de exemplos (metade positivos, metade negativos).
    max_words_per_example : int | None, optional
        Trunca cada exemplo para no máximo esta quantidade de palavras, by
        default None.
    random_seed : int | None, optional
        Semente para a amostragem aleatória dos exemplos negativos, by
        default None.

    Returns
    -------
    dict[str, list[str] | list[float]]
        Dicionário com ``positive_texts``, ``negative_texts``,
        ``positive_activations`` e ``negative_activations``.
    """
    rng = np.random.default_rng(random_seed)

    neuron_activations = activations[:, neuron_idx]
    n_per_class = n_examples // 2

    n_positive = int(np.sum(neuron_activations > 0))
    if n_positive < n_per_class:
        logger.warning(
            "Apenas %d exemplo(s) com ativação positiva para o neurônio %d; "
            "usando todos os disponíveis",
            n_positive,
            neuron_idx,
        )
        top_indices = np.argsort(neuron_activations)[-n_positive:]
    else:
        top_indices = np.argsort(neuron_activations)[-n_per_class:]

    zero_indices = np.where(neuron_activations == 0)[0]
    if len(zero_indices) >= n_per_class:
        random_indices = rng.choice(zero_indices, size=n_per_class, replace=False)
    else:
        logger.warning(
            "Apenas %d exemplo(s) com ativação zero para o neurônio %d; "
            "usando todos os disponíveis",
            len(zero_indices),
            neuron_idx,
        )
        random_indices = zero_indices

    positive_texts = [texts[i] for i in top_indices]
    negative_texts = [texts[i] for i in random_indices]

    if max_words_per_example:
        positive_texts = [truncate_text(text, max_words_per_example) for text in positive_texts]
        negative_texts = [truncate_text(text, max_words_per_example) for text in negative_texts]

    return {
        "positive_texts": positive_texts,
        "negative_texts": negative_texts,
        "positive_activations": neuron_activations[top_indices].tolist(),
        "negative_activations": neuron_activations[random_indices].tolist(),
    }


def sample_percentile_bins(
    texts: list[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: int | None = None,
    high_percentile: tuple[float, float] = (90, 100),
    low_percentile: tuple[float, float] | None = None,
    random_seed: int | None = None,
) -> dict[str, list[str] | list[float]]:
    """Amostra exemplos de uma faixa de percentil alto e de outra baixa (ou ativação zero).

    Parameters
    ----------
    texts : list[str]
        Todos os textos do corpus.
    activations : np.ndarray
        Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
    neuron_idx : int
        Índice do neurônio a amostrar.
    n_examples : int
        Número total de exemplos (metade positivos, metade negativos).
    max_words_per_example : int | None, optional
        Trunca cada exemplo para no máximo esta quantidade de palavras, by
        default None.
    high_percentile : tuple[float, float], optional
        Faixa de percentil (ativações positivas) usada nos exemplos
        positivos, by default (90, 100).
    low_percentile : tuple[float, float] | None, optional
        Faixa de percentil usada nos exemplos negativos; se ``None``, usa
        ativações exatamente zero, by default None.
    random_seed : int | None, optional
        Semente para a amostragem aleatória dentro de cada faixa, by
        default None.

    Returns
    -------
    dict[str, list[str] | list[float]]
        Dicionário com ``positive_texts``, ``negative_texts``,
        ``positive_activations`` e ``negative_activations``.
    """
    rng = np.random.default_rng(random_seed)

    neuron_activations = activations[:, neuron_idx]
    n_per_class = n_examples // 2

    positive_mask = neuron_activations > 0
    positive_values = neuron_activations[positive_mask]
    positive_indices = np.where(positive_mask)[0]

    high_mask = (positive_values >= np.percentile(positive_values, high_percentile[0])) & (
        positive_values <= np.percentile(positive_values, high_percentile[1])
    )
    high_indices = positive_indices[high_mask]
    if len(high_indices) >= n_per_class:
        high_sample_indices = rng.choice(high_indices, size=n_per_class, replace=False)
    else:
        logger.warning(
            "Menos de %d exemplo(s) na faixa %s para o neurônio %d; usando %d",
            n_per_class,
            high_percentile,
            neuron_idx,
            len(high_indices),
        )
        high_sample_indices = high_indices

    if low_percentile is not None:
        low_mask = (positive_values >= np.percentile(positive_values, low_percentile[0])) & (
            positive_values <= np.percentile(positive_values, low_percentile[1])
        )
        low_indices = positive_indices[low_mask]
    else:
        low_indices = np.where(neuron_activations == 0)[0]

    if len(low_indices) >= n_per_class:
        low_sample_indices = rng.choice(low_indices, size=n_per_class, replace=False)
    else:
        logger.warning(
            "Menos de %d exemplo(s) na faixa %s para o neurônio %d; usando %d",
            n_per_class,
            low_percentile,
            neuron_idx,
            len(low_indices),
        )
        low_sample_indices = low_indices

    positive_texts = [texts[i] for i in high_sample_indices]
    negative_texts = [texts[i] for i in low_sample_indices]

    if max_words_per_example:
        positive_texts = [truncate_text(text, max_words_per_example) for text in positive_texts]
        negative_texts = [truncate_text(text, max_words_per_example) for text in negative_texts]

    return {
        "positive_texts": positive_texts,
        "negative_texts": negative_texts,
        "positive_activations": neuron_activations[high_sample_indices].tolist(),
        "negative_activations": neuron_activations[low_sample_indices].tolist(),
    }


@dataclass
class SamplingConfig:
    """Configuração de amostragem de exemplos para o prompt de interpretação.

    Parameters
    ----------
    function : Callable
        Função de amostragem, by default :func:`sample_top_zero`.
    n_examples : int
        Número de exemplos a amostrar para o prompt do intérprete, by
        default 20.
    random_seed : int | None
        Semente base; cada candidato de interpretação incrementa esta
        semente em 1, by default 0.
    max_words_per_example : int | None
        Número máximo de palavras por exemplo, truncado se necessário, by
        default 256.
    sampling_kwargs : dict[str, Any]
        Argumentos extras repassados à função de amostragem.
    """

    function: Callable = sample_top_zero
    n_examples: int = 20
    random_seed: int | None = 0
    max_words_per_example: int | None = 256
    sampling_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """Configuração do LLM intérprete.

    Parameters
    ----------
    temperature : float | None
        Temperatura do modelo intérprete, by default None.
    max_output_tokens : int | None
        Máximo de tokens de saída por interpretação gerada, by default None.
    max_interpretation_tokens : int | None
        Alias retrocompatível de ``max_output_tokens``, by default None.
    timeout : float | None
        Timeout opcional da requisição, em segundos, by default None.
    reasoning_effort : str | None
        Esforço de raciocínio (modelos compatíveis), by default None.
    verbosity : str | None
        Verbosidade da resposta (modelos compatíveis), by default None.
    llm_kwargs : dict[str, Any]
        Argumentos extras repassados a :func:`~hypothesaes.llm_api.generate_completion`.
    """

    temperature: float | None = None
    max_output_tokens: int | None = None
    max_interpretation_tokens: int | None = None
    timeout: float | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpretConfig:
    """Configuração completa de uma rodada de interpretação de neurônios.

    Parameters
    ----------
    sampling : SamplingConfig
        Configuração de amostragem de exemplos.
    llm : LLMConfig
        Configuração do LLM intérprete.
    n_candidates : int
        Número de interpretações candidatas por neurônio, by default 1.
    interpretation_prompt_name : str
        Nome do template de prompt em ``prompts/``, by default
        "interpret-neuron-binary".
    task_specific_instructions : str
        Instruções específicas da tarefa, incluídas no prompt de
        interpretação.
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    n_candidates: int = 1
    interpretation_prompt_name: str = "interpret-neuron-binary"
    task_specific_instructions: str = DEFAULT_TASK_SPECIFIC_INSTRUCTIONS


@dataclass
class ScoringConfig:
    """Configuração da pontuação de fidelidade das interpretações.

    Parameters
    ----------
    n_examples : int
        Número de exemplos usados para pontuar a fidelidade (metade
        ativando fortemente, metade sem ativação), by default 100.
    max_words_per_example : int | None
        Número máximo de palavras por exemplo, by default 256.
    sampling_function : Callable
        Função de amostragem dos exemplos de pontuação, by default
        :func:`sample_top_zero`.
    sampling_kwargs : dict[str, Any]
        Argumentos extras repassados à função de amostragem.
    """

    n_examples: int = 100
    max_words_per_example: int | None = 256
    sampling_function: Callable = sample_top_zero
    sampling_kwargs: dict[str, Any] = field(default_factory=dict)


class NeuronInterpreter:
    """Gera e pontua interpretações em linguagem natural para neurônios do SAE.

    Parameters
    ----------
    interpreter_model : str, optional
        Modelo LLM usado para gerar interpretações, by default "gpt-5.2".
    annotator_model : str, optional
        Modelo LLM usado para pontuar a fidelidade das interpretações, by
        default "gpt-5-mini".
    n_workers_interpretation : int, optional
        Número de threads paralelas para geração de interpretações, by
        default 10.
    n_workers_annotation : int, optional
        Número de threads paralelas para anotação (pontuação), by default
        30.
    cache_name : str | None, optional
        Nome do cache de anotações usado na pontuação de fidelidade, by
        default None.
    """

    def __init__(
        self,
        interpreter_model: str = "gpt-5.2",
        annotator_model: str = "gpt-5-mini",
        n_workers_interpretation: int = 10,
        n_workers_annotation: int = 30,
        cache_name: str | None = None,
    ) -> None:
        self.interpreter_model = interpreter_model
        self.annotator_model = annotator_model
        self.n_workers_interpretation = n_workers_interpretation
        self.n_workers_annotation = n_workers_annotation
        self.cache_name = cache_name

    def _build_interpretation_prompt(
        self,
        texts: list[str],
        activations: np.ndarray,
        neuron_idx: int,
        candidate_idx: int,
        config: InterpretConfig,
    ) -> str | None:
        """Monta o prompt de interpretação para um neurônio, ou ``None`` se ele estiver morto."""
        if np.all(activations[:, neuron_idx] <= 0):
            logger.warning(
                "Todas as ativações do neurônio %d são <= 0. Este neurônio pode estar morto; "
                "pulando interpretação.",
                neuron_idx,
            )
            return None

        formatted_examples = config.sampling.function(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            n_examples=config.sampling.n_examples,
            max_words_per_example=config.sampling.max_words_per_example,
            random_seed=(
                config.sampling.random_seed + candidate_idx
                if config.sampling.random_seed is not None
                else None
            ),
            **config.sampling.sampling_kwargs,
        )

        try:
            interpretation_prompt_template = load_prompt_template(config.interpretation_prompt_name)
            return interpretation_prompt_template.format(
                task_specific_instructions=config.task_specific_instructions, **formatted_examples
            )
        except KeyError as exception:
            raise KeyError(
                f"Chave obrigatória {exception} ausente no template de interpretação. "
                "Garanta que todas as chaves esperadas estejam em formatted_examples."
            ) from exception

    def _parse_interpretation(self, response: str) -> str | None:
        """Interpreta a resposta bruta do LLM, retornando o texto limpo da interpretação."""
        response = response.strip()

        # Resposta incompleta (o modelo começou a "pensar" mas não terminou).
        if "<think>" in response and "</think>" not in response:
            return None
        if "</think>" in response:
            response = response.split("</think>")[1].strip()

        response = response.split("\n", 1)[0]
        for prefix in ("- ", '"-', '" -'):
            response = response.removeprefix(prefix)

        return response.strip('"').strip()

    def _resolve_llm_kwargs(
        self, config: InterpretConfig, llm_kwargs: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], float | None]:
        """Resolve os argumentos da requisição a partir dos padrões da config e de overrides."""
        default_max_output_tokens = config.llm.max_output_tokens
        if default_max_output_tokens is None:
            default_max_output_tokens = config.llm.max_interpretation_tokens

        merged_kwargs = config.llm.llm_kwargs.copy()
        if llm_kwargs:
            merged_kwargs.update(llm_kwargs)

        resolved = normalize_llm_kwargs(
            merged_kwargs,
            default_verbosity=config.llm.verbosity,
            default_reasoning_effort=config.llm.reasoning_effort,
            default_timeout=config.llm.timeout,
            default_max_output_tokens=default_max_output_tokens,
        )
        timeout = resolved.pop("timeout", None)
        if "temperature" not in resolved and config.llm.temperature is not None:
            resolved["temperature"] = config.llm.temperature
        return resolved, timeout

    def _generate_interpretation(
        self, prompt: str, llm_kwargs: dict[str, Any] | None = None, timeout: float | None = None
    ) -> str | None:
        """Envia um único prompt ao modelo intérprete e retorna a interpretação já parseada."""
        try:
            request_kwargs = dict(llm_kwargs or {})
            if timeout is not None:
                request_kwargs["timeout"] = timeout
            response = generate_completion(
                prompt=prompt, model=self.interpreter_model, **request_kwargs
            )
            return self._parse_interpretation(response)
        except Exception:
            logger.exception("Falha ao gerar interpretação.")
            return None

    def _execute_prompts(
        self, prompts: list[str | None], llm_kwargs: dict[str, Any], timeout: float | None
    ) -> list[str | None]:
        """Executa um lote de prompts em paralelo e retorna as interpretações (uma por prompt)."""
        valid_prompts = [prompt for prompt in prompts if prompt is not None]
        if not valid_prompts:
            return []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.n_workers_interpretation
        ) as executor:
            future_to_idx = {
                executor.submit(
                    self._generate_interpretation, prompt, llm_kwargs=llm_kwargs, timeout=timeout
                ): idx
                for idx, prompt in enumerate(valid_prompts)
            }

            iterator = tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(valid_prompts),
                desc="Gerando interpretações",
            )

            ordered_interpretations: list[str | None] = [None] * len(valid_prompts)
            for future in iterator:
                ordered_interpretations[future_to_idx[future]] = future.result()
            return ordered_interpretations

    def interpret_neurons(
        self,
        texts: list[str],
        activations: np.ndarray,
        neuron_indices: list[int],
        config: InterpretConfig | None = None,
        **llm_kwargs: Any,
    ) -> dict[int, list[str | None]]:
        """Gera interpretações para múltiplos neurônios, com múltiplos candidatos cada.

        Parameters
        ----------
        texts : list[str]
            Todos os textos do corpus.
        activations : np.ndarray
            Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
        neuron_indices : list[int]
            Índices dos neurônios a interpretar.
        config : InterpretConfig | None, optional
            Configuração de amostragem/LLM, by default None (usa
            :class:`InterpretConfig` padrão).
        **llm_kwargs : Any
            Argumentos extras repassados à requisição do LLM.

        Returns
        -------
        dict[int, list[str | None]]
            Mapa de índice de neurônio para a lista de interpretações
            candidatas geradas (``None`` para candidatos que falharam).
        """
        config = config or InterpretConfig()
        resolved_llm_kwargs, timeout = self._resolve_llm_kwargs(config, llm_kwargs)
        interpretation_tasks = [
            (neuron_idx, candidate_idx)
            for neuron_idx in neuron_indices
            for candidate_idx in range(config.n_candidates)
        ]

        interpretations: dict[int, list[str | None]] = {idx: [] for idx in neuron_indices}

        prompts = [
            self._build_interpretation_prompt(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                candidate_idx=candidate_idx,
                config=config,
            )
            for neuron_idx, candidate_idx in interpretation_tasks
        ]

        generated_interpretations = iter(
            self._execute_prompts(prompts, resolved_llm_kwargs, timeout)
        )

        for idx, (neuron_idx, _candidate_idx) in enumerate(interpretation_tasks):
            if prompts[idx] is None:
                interpretations[neuron_idx].append(None)
            else:
                interpretations[neuron_idx].append(next(generated_interpretations))

        return interpretations

    def _compute_metrics(
        self, annotations: np.ndarray, labels: np.ndarray, activations: np.ndarray
    ) -> dict[str, float]:
        """Calcula métricas de avaliação de uma única interpretação.

        Parameters
        ----------
        annotations : np.ndarray
            Anotações geradas por um LLM ao aplicar a interpretação em
            linguagem natural do neurônio a um conjunto de exemplos.
        labels : np.ndarray
            Ativações do neurônio binarizadas (ex.: top-N ativações = 1,
            ativações zero = 0) para os exemplos pontuados.
        activations : np.ndarray
            Ativações contínuas do neurônio para os exemplos pontuados.

        Returns
        -------
        dict[str, float]
            Dicionário com ``recall``, ``precision``, ``f1`` e ``correlation``.
        """
        if not (1 in labels and 0 in labels):
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}

        annotations = np.asarray(annotations).astype(bool)
        labels = np.asarray(labels).astype(bool)

        true_positives = np.sum(annotations & labels)
        false_positives = np.sum(annotations & ~labels)
        false_negatives = np.sum(~annotations & labels)

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        correlation = (
            np.corrcoef(activations, annotations)[0, 1] if len(np.unique(annotations)) > 1 else 0.0
        )

        return {"recall": recall, "precision": precision, "f1": f1, "correlation": correlation}

    def _build_scoring_tasks(
        self,
        texts: list[str],
        activations: np.ndarray,
        interpretations: Mapping[int, Sequence[str | None]],
        config: ScoringConfig,
    ) -> tuple[list[tuple[str, str]], dict[int, dict[str, Any]]]:
        """Monta as tarefas de anotação e as informações de amostragem por neurônio."""
        tasks: list[tuple[str, str]] = []
        scoring_info: dict[int, dict[str, Any]] = {}

        for neuron_idx, neuron_interpretations in interpretations.items():
            formatted_examples = config.sampling_function(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                n_examples=config.n_examples,
                max_words_per_example=config.max_words_per_example,
                random_seed=neuron_idx,
                **config.sampling_kwargs,
            )

            eval_texts = formatted_examples["positive_texts"] + formatted_examples["negative_texts"]
            scoring_info[neuron_idx] = {
                "texts": eval_texts,
                "activations": formatted_examples["positive_activations"]
                + formatted_examples["negative_activations"],
                "binarized_activations": np.concatenate(
                    [
                        np.ones(len(formatted_examples["positive_texts"])),
                        np.zeros(len(formatted_examples["negative_texts"])),
                    ]
                ),
            }

            tasks.extend(
                (text, interpretation)
                for interpretation in neuron_interpretations
                if interpretation is not None
                for text in eval_texts
            )

        return tasks, scoring_info

    def _score_single_interpretation(
        self,
        interpretation: str | None,
        neuron_scoring_info: dict[str, Any],
        annotations: dict[str, dict[str, int]],
    ) -> dict[str, float]:
        """Calcula as métricas de fidelidade de uma interpretação candidata (0 se ela falhou)."""
        if interpretation is None:
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}

        annotation_values = [
            annotations[interpretation][text] for text in neuron_scoring_info["texts"]
        ]
        return self._compute_metrics(
            annotations=np.array(annotation_values),
            labels=neuron_scoring_info["binarized_activations"],
            activations=neuron_scoring_info["activations"],
        )

    def _compile_metrics(
        self,
        interpretations: Mapping[int, Sequence[str | None]],
        scoring_info: dict[int, dict[str, Any]],
        annotations: dict[str, dict[str, int]],
    ) -> dict[int, dict[str | None, dict[str, float]]]:
        """Compila as métricas de fidelidade de cada interpretação a partir das anotações."""
        all_metrics: dict[int, dict[str | None, dict[str, float]]] = {}
        for neuron_idx, neuron_interpretations in interpretations.items():
            neuron_scoring_info = scoring_info[neuron_idx]
            all_metrics[neuron_idx] = {
                interpretation: self._score_single_interpretation(
                    interpretation, neuron_scoring_info, annotations
                )
                for interpretation in neuron_interpretations
            }
        return all_metrics

    def score_interpretations(
        self,
        texts: list[str],
        activations: np.ndarray,
        interpretations: Mapping[int, Sequence[str | None]],
        config: ScoringConfig | None = None,
        show_progress: bool = True,
        **annotation_kwargs: Any,
    ) -> dict[int, dict[str | None, dict[str, float]]]:
        """Pontua a fidelidade de todas as interpretações de todos os neurônios.

        Parameters
        ----------
        texts : list[str]
            Todos os textos do corpus.
        activations : np.ndarray
            Matriz de ativações do SAE, formato ``(n_amostras, n_neuronios)``.
        interpretations : Mapping[int, Sequence[str | None]]
            Interpretações candidatas por neurônio (ver :meth:`interpret_neurons`).
        config : ScoringConfig | None, optional
            Configuração de amostragem para a pontuação, by default None.
        show_progress : bool, optional
            Se exibe barra de progresso, by default True.
        **annotation_kwargs : Any
            Argumentos extras repassados a :func:`~hypothesaes.annotate.annotate_tasks`.

        Returns
        -------
        dict[int, dict[str | None, dict[str, float]]]
            Mapa aninhado ``{neuron_idx: {interpretação: métricas}}``.
        """
        config = config or ScoringConfig()
        tasks, scoring_info = self._build_scoring_tasks(texts, activations, interpretations, config)

        cache_path = (
            None
            if self.cache_name is None
            else ANNOTATION_CACHE_DIR / f"{self.cache_name}_interp-scoring.json"
        )
        n_candidates = len(next(iter(interpretations.values())))
        progress_desc = (
            f"Pontuando fidelidade das interpretações ({len(interpretations)} neurônio(s); "
            f"{n_candidates} candidato(s) por neurônio; "
            f"{config.n_examples} exemplo(s) por interpretação)"
        )
        annotations = annotate_tasks(
            tasks=tasks,
            cache_path=cache_path,
            n_workers=self.n_workers_annotation,
            show_progress=show_progress,
            model=self.annotator_model,
            progress_desc=progress_desc,
            **annotation_kwargs,
        )

        return self._compile_metrics(interpretations, scoring_info, annotations)
