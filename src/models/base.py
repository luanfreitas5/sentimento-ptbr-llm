"""Interface comum e blocos de construção compartilhados entre os modelos de sentimento.

Implementa a Fase 9 do plano de elaboração (``PLANO-ELABORACAO.md``) e as
Seções 4.4-4.5 do documento mestre (``projeto-mestrado-analise-sentimentos-ptbr.md``):
define o contrato ``fit``/``predict``/``predict_proba`` que todo classificador
de sentimento do projeto (clássico, profundo, Transformer ou LLM) deve
satisfazer, permitindo que ``src/models/factory.py``, ``src/training`` e
``src/inference`` operem sobre qualquer modelo sem conhecer sua implementação
concreta.

Também concentra duas implementações genéricas reaproveitadas pelos modelos
concretos, para evitar duplicar a mesma engenharia em múltiplos arquivos:

- :class:`TransformerSentimentClassifier` - motor de fine-tuning via
  ``transformers``, parametrizado pelo nome do modelo pré-treinado e por
  hiperparâmetros; ``bertimbau.py``, ``roberta.py`` e ``distilbert.py`` são
  fábricas finas sobre esta classe.
- :func:`build_token_vocabulary` / :func:`encode_token_sequences` -
  utilitários de vocabulário e padding compartilhados por ``lstm.py`` e
  ``cnn.py``.
"""

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from exceptions.model import ModelError, ModelNotFittedError
from utils.seed import seed_everything
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@runtime_checkable
class SentimentClassifier(Protocol):
    """Interface mínima que todo classificador de sentimento do projeto deve implementar.

    Modelada sobre a API de estimadores do scikit-learn (satisfeita por
    duck typing pelos classificadores clássicos de
    ``src/models/naive_bayes.py`` etc.) e implementada explicitamente pelos
    modelos deste módulo (:class:`TransformerSentimentClassifier`) e por
    ``src/models/lstm.py``, ``src/models/cnn.py`` e ``src/models/llm.py``.
    """

    def fit(self, X: Any, y: Any) -> "SentimentClassifier":
        """Treina o classificador.

        Parameters
        ----------
        X : Any
            Dados de entrada (matriz de features, documentos tokenizados ou
            textos crus, conforme o modelo concreto).
        y : Any
            Rótulos de sentimento de treino, pertencentes a
            :data:`constants.labels.SENTIMENT_CLASSES`.

        Returns
        -------
        SentimentClassifier
            A própria instância, treinada.
        """
        ...

    def predict(self, X: Any) -> np.ndarray:
        """Prediz o rótulo de sentimento mais provável para cada amostra.

        Parameters
        ----------
        X : Any
            Dados de entrada, no mesmo formato usado em :meth:`fit`.

        Returns
        -------
        np.ndarray
            Vetor de rótulos de sentimento preditos, um por amostra.
        """
        ...

    def predict_proba(self, X: Any) -> np.ndarray:
        """Estima a distribuição de probabilidade por classe de sentimento.

        Parameters
        ----------
        X : Any
            Dados de entrada, no mesmo formato usado em :meth:`fit`.

        Returns
        -------
        np.ndarray
            Matriz ``(n_amostras, n_classes)`` de probabilidades.
        """
        ...


def build_token_vocabulary(
    tokenized_documents: Sequence[Sequence[str]],
    *,
    max_vocabulary_size: int | None = None,
    minimum_document_frequency: int = 1,
) -> dict[str, int]:
    """Constrói um vocabulário termo -> índice a partir de documentos tokenizados.

    Reserva o índice ``0`` para :data:`PAD_TOKEN` (padding) e o índice ``1``
    para :data:`UNK_TOKEN` (termos fora do vocabulário), convenção usada por
    :func:`encode_token_sequences` e pelas camadas ``nn.Embedding`` de
    ``src/models/lstm.py``/``src/models/cnn.py``.

    Parameters
    ----------
    tokenized_documents : Sequence[Sequence[str]]
        Um documento por elemento, já tokenizado (ex.: saída de
        ``src/preprocessing/tokenization.py``). Não vazio.
    max_vocabulary_size : int | None, optional
        Número máximo de termos (além de ``PAD``/``UNK``), mantendo os mais
        frequentes. Sem limite quando ``None``, by default None.
    minimum_document_frequency : int, optional
        Frequência de documento mínima para um termo entrar no vocabulário,
        by default 1.

    Returns
    -------
    dict[str, int]
        Vocabulário termo -> índice, incluindo ``PAD_TOKEN`` e ``UNK_TOKEN``.

    Raises
    ------
    EmptyDatasetError
        Se ``tokenized_documents`` estiver vazio.

    Examples
    --------
    >>> vocabulario = build_token_vocabulary([["bom", "dia"], ["bom", "produto"]])
    >>> vocabulario["<pad>"], vocabulario["<unk>"]
    (0, 1)
    >>> vocabulario["bom"]
    2
    """
    validate_not_empty_collection(tokenized_documents, collection_name="tokenized_documents")

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    eligible_terms = [
        term
        for term, frequency in document_frequency.items()
        if frequency >= minimum_document_frequency
    ]
    eligible_terms.sort(key=lambda term: (-document_frequency[term], term))
    if max_vocabulary_size is not None:
        eligible_terms = eligible_terms[:max_vocabulary_size]

    vocabulary: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for term in eligible_terms:
        vocabulary[term] = len(vocabulary)
    return vocabulary


def encode_token_sequences(
    tokenized_documents: Sequence[Sequence[str]],
    vocabulary: Mapping[str, int],
    *,
    max_sequence_length: int,
) -> np.ndarray:
    """Codifica documentos tokenizados em sequências de índices inteiros.

    Trunca documentos mais longos que ``max_sequence_length`` e preenche
    (``padding``) os mais curtos com o índice de :data:`PAD_TOKEN`.

    Parameters
    ----------
    tokenized_documents : Sequence[Sequence[str]]
        Um documento por elemento, já tokenizado. Não vazio.
    vocabulary : Mapping[str, int]
        Vocabulário termo -> índice, via :func:`build_token_vocabulary`.
    max_sequence_length : int
        Comprimento fixo de saída (em tokens) de cada sequência codificada.

    Returns
    -------
    np.ndarray
        Matriz ``(n_documentos, max_sequence_length)`` de índices inteiros
        (``dtype=int64``), pronta para uma camada ``nn.Embedding``.

    Raises
    ------
    EmptyDatasetError
        Se ``tokenized_documents`` estiver vazio.

    Examples
    --------
    >>> vocabulario = {"<pad>": 0, "<unk>": 1, "bom": 2, "dia": 3}
    >>> encode_token_sequences([["bom", "dia"]], vocabulario, max_sequence_length=3)
    array([[2, 3, 0]], dtype=int64)
    """
    validate_not_empty_collection(tokenized_documents, collection_name="tokenized_documents")

    pad_index = vocabulary[PAD_TOKEN]
    unknown_index = vocabulary[UNK_TOKEN]
    encoded = np.full((len(tokenized_documents), max_sequence_length), pad_index, dtype=np.int64)
    for row_index, tokens in enumerate(tokenized_documents):
        for column_index, token in enumerate(tokens[:max_sequence_length]):
            encoded[row_index, column_index] = vocabulary.get(token, unknown_index)
    return encoded


class TransformerSentimentClassifier:
    """Classificador de sentimento por fine-tuning de um encoder Transformer pré-treinado.

    Motor genérico reaproveitado por ``src/models/bertimbau.py``,
    ``src/models/roberta.py`` e ``src/models/distilbert.py``: cada um
    fornece apenas o nome do modelo pré-treinado e os hiperparâmetros
    específicos (``configs/model_params.yaml -> transformers.*``); a lógica
    de tokenização, treino (com parada antecipada sobre 10% dos dados
    reservados para validação, como em ``src/features/reduction.py``) e
    inferência é idêntica entre eles.

    ``torch``/``transformers`` são dependências pesadas e opcionais: o
    import ocorre de forma tardia, em :meth:`fit`/:meth:`predict_proba`,
    para que este módulo permaneça importável sem elas.

    Parameters
    ----------
    model_name : str
        Nome do modelo pré-treinado no Hugging Face Hub.
    max_length : int, optional
        Comprimento máximo de subtokens por texto, by default 128.
    batch_size : int, optional
        Tamanho do lote de treino, by default 16.
    learning_rate : float, optional
        Taxa de aprendizado do otimizador AdamW, by default 0.00002.
    epochs : int, optional
        Número máximo de épocas de fine-tuning, by default 4.
    warmup_ratio : float, optional
        Fração dos passos de treino usada para aquecimento linear da taxa
        de aprendizado, by default 0.1.
    weight_decay : float, optional
        Decaimento de peso (regularização L2) do otimizador, by default 0.01.
    early_stopping_patience : int, optional
        Épocas sem melhora na perda de validação antes de interromper o
        treino, by default 2.
    random_state : int, optional
        Semente aleatória, by default 42.
    device : str | None, optional
        Dispositivo PyTorch (``"cpu"``, ``"cuda"``). Se ``None``, usa
        ``"cuda"`` quando disponível e ``"cpu"`` caso contrário, by default
        None.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 0.00002,
        epochs: int = 4,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 2,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = device
        self.classes_: np.ndarray | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str | None = None

    def fit(self, X: Sequence[str], y: Sequence[str]) -> "TransformerSentimentClassifier":
        """Executa o fine-tuning do encoder sobre os textos e rótulos de treino.

        Parameters
        ----------
        X : Sequence[str]
            Textos de treino, não vazio.
        y : Sequence[str]
            Rótulos de sentimento de treino, mesmo tamanho de ``X``.

        Returns
        -------
        TransformerSentimentClassifier
            A própria instância, com o modelo ajustado.

        Raises
        ------
        EmptyDatasetError
            Se ``X`` estiver vazio.
        ModelError
            Se ``torch``/``transformers`` não estiverem instalados.

        Examples
        --------
        >>> TransformerSentimentClassifier("modelo-de-exemplo").fit(
        ...     ["ótimo produto", "péssimo atendimento"], ["positivo", "negativo"]
        ... )  # doctest: +SKIP
        """
        validate_not_empty_collection(X, collection_name="X")
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
            )
        except ImportError as exception:
            raise ModelError(
                "As bibliotecas 'transformers'/'torch' não estão instaladas. Instale com "
                "`uv add transformers torch` (ou `uv sync --extra llm`) para o fine-tuning "
                f"de '{self.model_name}'."
            ) from exception

        seed_everything(self.random_state)
        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.classes_ = np.array(sorted(set(y)))
        label_to_index = {label: index for index, label in enumerate(self.classes_)}
        encoded_labels = torch.tensor([label_to_index[label] for label in y], dtype=torch.long)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes_)
        ).to(resolved_device)

        encoded_input = self._tokenizer(
            list(X), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        validation_size = max(1, int(0.1 * len(X)))
        permutation = np.random.default_rng(self.random_state).permutation(len(X))
        validation_indices = permutation[:validation_size]
        train_indices = permutation[validation_size:]

        train_dataset = TensorDataset(
            encoded_input["input_ids"][train_indices],
            encoded_input["attention_mask"][train_indices],
            encoded_labels[train_indices],
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        validation_input_ids = encoded_input["input_ids"][validation_indices].to(resolved_device)
        validation_attention_mask = encoded_input["attention_mask"][validation_indices].to(
            resolved_device
        )
        validation_labels = encoded_labels[validation_indices].to(resolved_device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_steps = max(len(train_loader) * self.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self._model.train()
            for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self._model(
                    input_ids=batch_input_ids.to(resolved_device),
                    attention_mask=batch_attention_mask.to(resolved_device),
                    labels=batch_labels.to(resolved_device),
                )
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()

            self._model.eval()
            with torch.no_grad():
                validation_loss = self._model(
                    input_ids=validation_input_ids,
                    attention_mask=validation_attention_mask,
                    labels=validation_labels,
                ).loss.item()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info("Parada antecipada na época %d.", epoch)
                break

        self._device = resolved_device
        logger.info(
            "Fine-tuning de '%s' concluído: %d classe(s), perda de validação final %.4f.",
            self.model_name,
            len(self.classes_),
            best_validation_loss,
        )
        return self

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Estima a distribuição de probabilidade por classe de sentimento.

        Parameters
        ----------
        X : Sequence[str]
            Textos a classificar.

        Returns
        -------
        np.ndarray
            Matriz ``(len(X), n_classes)`` de probabilidades.

        Raises
        ------
        ModelNotFittedError
            Se o modelo ainda não tiver sido treinado via :meth:`fit`.
        """
        if self._model is None or self._tokenizer is None:
            raise ModelNotFittedError(self.model_name)
        import torch

        encoded_input = self._tokenizer(
            list(X), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**encoded_input).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Prediz o rótulo de sentimento mais provável para cada texto.

        Parameters
        ----------
        X : Sequence[str]
            Textos a classificar.

        Returns
        -------
        np.ndarray
            Vetor de rótulos de sentimento preditos.

        Raises
        ------
        ModelNotFittedError
            Se o modelo ainda não tiver sido treinado via :meth:`fit`.
        """
        probabilities = self.predict_proba(X)
        assert self.classes_ is not None  # garantido por predict_proba não ter levantado exceção
        return self.classes_[probabilities.argmax(axis=1)]
