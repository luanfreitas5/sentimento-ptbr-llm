"""Classificador de sentimento por LSTM/BiLSTM (PyTorch) sobre texto tokenizado.

Implementa a Fase 9 (Seção 4.5 do documento mestre): uma camada de embedding
treinada do zero seguida de uma (Bi)LSTM e uma cabeça linear de
classificação (``configs/model_params.yaml -> deep_learning.recurrent``).
Diferente dos classificadores clássicos (``src/models/logistic_regression.py``
etc.), não depende de features pré-computadas - recebe diretamente os
documentos tokenizados por ``src/preprocessing/tokenization.py``.

``torch`` é uma dependência pesada e opcional: o import ocorre de forma
tardia, em :meth:`LSTMSentimentClassifier.fit`/:meth:`predict_proba`, para
que este módulo permaneça importável sem ela.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from exceptions.model import ModelError, ModelNotFittedError
from models.base import build_token_vocabulary, encode_token_sequences
from utils.seed import seed_everything
from utils.validation import validate_not_empty_collection

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class LSTMSentimentClassifier:
    """Classificador de sentimento por (Bi)LSTM sobre embeddings treinados do zero.

    Parameters
    ----------
    embedding_dim : int, optional
        Dimensão da camada de embedding treinada do zero, by default 300.
    hidden_dim : int, optional
        Dimensão do estado oculto da LSTM, by default 128.
    num_layers : int, optional
        Número de camadas empilhadas da LSTM, by default 2.
    bidirectional : bool, optional
        Se ``True``, usa uma BiLSTM, by default True.
    dropout : float, optional
        Taxa de dropout entre camadas da LSTM e antes da cabeça linear, by
        default 0.3.
    batch_size : int, optional
        Tamanho do lote de treino, by default 32.
    epochs : int, optional
        Número máximo de épocas, by default 30.
    learning_rate : float, optional
        Taxa de aprendizado do otimizador Adam, by default 0.001.
    early_stopping_patience : int, optional
        Épocas sem melhora na perda de validação antes de interromper o
        treino, by default 5.
    max_sequence_length : int, optional
        Comprimento máximo de tokens por documento (truncamento/padding),
        by default 128.
    max_vocabulary_size : int | None, optional
        Tamanho máximo do vocabulário, sem limite quando ``None``, by
        default None.
    random_state : int, optional
        Semente aleatória, by default 42.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
        max_sequence_length: int = 128,
        max_vocabulary_size: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.max_sequence_length = max_sequence_length
        self.max_vocabulary_size = max_vocabulary_size
        self.random_state = random_state
        self.classes_: np.ndarray | None = None
        self.vocabulary_: dict[str, int] | None = None
        self._module: Any = None

    def _encode_training_data(
        self,
        X: Sequence[Sequence[str]],  # noqa: N803
        y: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Constrói o vocabulário/rótulos e codifica ``X``/``y`` para tensores de treino.

        Parameters
        ----------
        X : Sequence[Sequence[str]]
            Documentos de treino, já tokenizados.
        y : Sequence[str]
            Rótulos de sentimento de treino, mesmo tamanho de ``X``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int, int]
            Sequências codificadas, rótulos codificados, tamanho do
            vocabulário e número de classes, nesta ordem.
        """
        self.vocabulary_ = build_token_vocabulary(X, max_vocabulary_size=self.max_vocabulary_size)
        encoded_sequences = encode_token_sequences(
            X, self.vocabulary_, max_sequence_length=self.max_sequence_length
        )
        self.classes_ = np.array(sorted(set(y)))
        label_to_index = {label: index for index, label in enumerate(self.classes_)}
        encoded_labels = np.array([label_to_index[label] for label in y], dtype=np.int64)
        return encoded_sequences, encoded_labels, len(self.vocabulary_), len(self.classes_)

    def _split_train_validation(
        self,
        torch: Any,
        data_loader_cls: Any,
        tensor_dataset_cls: Any,
        encoded_sequences: np.ndarray,
        encoded_labels: np.ndarray,
    ) -> tuple[Any, "torch.Tensor", "torch.Tensor"]:
        """Separa 10% das amostras para validação e monta o ``DataLoader`` de treino.

        Parameters
        ----------
        torch : Any
            Módulo ``torch`` importado tardiamente por :meth:`fit`.
        data_loader_cls : Any
            Classe ``torch.utils.data.DataLoader``.
        tensor_dataset_cls : Any
            Classe ``torch.utils.data.TensorDataset``.
        encoded_sequences : np.ndarray
            Sequências de índices de token codificadas.
        encoded_labels : np.ndarray
            Rótulos codificados, mesmo tamanho de ``encoded_sequences``.

        Returns
        -------
        tuple[Any, torch.Tensor, torch.Tensor]
            ``DataLoader`` de treino, tensor de validação e rótulos de
            validação, nesta ordem.
        """
        validation_size = max(1, int(0.1 * len(encoded_sequences)))
        permutation = np.random.default_rng(self.random_state).permutation(len(encoded_sequences))
        validation_indices = permutation[:validation_size]
        train_indices = permutation[validation_size:]

        train_tensor = torch.tensor(encoded_sequences[train_indices], dtype=torch.long)
        train_labels_tensor = torch.tensor(encoded_labels[train_indices], dtype=torch.long)
        validation_tensor = torch.tensor(encoded_sequences[validation_indices], dtype=torch.long)
        validation_labels_tensor = torch.tensor(
            encoded_labels[validation_indices], dtype=torch.long
        )

        train_loader = data_loader_cls(
            tensor_dataset_cls(train_tensor, train_labels_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_loader, validation_tensor, validation_labels_tensor

    def _run_training_loop(
        self,
        module: Any,
        torch: Any,
        optimizer: Any,
        loss_function: Any,
        train_loader: Any,
        validation_tensor: "torch.Tensor",
        validation_labels_tensor: "torch.Tensor",
    ) -> None:
        """Executa o laço de épocas com parada antecipada sobre a perda de validação.

        Parameters
        ----------
        module : Any
            Módulo PyTorch a ser treinado (mutado in-place).
        torch : Any
            Módulo ``torch`` importado tardiamente por :meth:`fit`.
        optimizer : Any
            Otimizador já associado aos parâmetros de ``module``.
        loss_function : Any
            Função de perda (ex.: ``nn.CrossEntropyLoss()``).
        train_loader : Any
            ``DataLoader`` com os lotes de treino.
        validation_tensor : torch.Tensor
            Tensor de entrada do conjunto de validação.
        validation_labels_tensor : torch.Tensor
            Rótulos do conjunto de validação.
        """
        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            module.train()
            for batch_sequences, batch_labels in train_loader:
                optimizer.zero_grad()
                loss = loss_function(module(batch_sequences), batch_labels)
                loss.backward()
                optimizer.step()

            module.eval()
            with torch.no_grad():
                validation_loss = loss_function(
                    module(validation_tensor), validation_labels_tensor
                ).item()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info("Parada antecipada na época %d.", epoch)
                break

    def fit(self, X: Sequence[Sequence[str]], y: Sequence[str]) -> "LSTMSentimentClassifier":  # noqa: N803
        """Treina a (Bi)LSTM sobre os documentos tokenizados de entrada.

        Constrói o vocabulário a partir de ``X``, reserva 10% das amostras
        para validação (parada antecipada), como em
        ``src/features/reduction.py``.

        Parameters
        ----------
        X : Sequence[Sequence[str]]
            Documentos de treino, já tokenizados. Não vazio.
        y : Sequence[str]
            Rótulos de sentimento de treino, mesmo tamanho de ``X``.

        Returns
        -------
        LSTMSentimentClassifier
            A própria instância, treinada.

        Raises
        ------
        EmptyDatasetError
            Se ``X`` estiver vazio.
        ModelError
            Se ``torch`` não estiver instalado.

        Examples
        --------
        >>> LSTMSentimentClassifier(epochs=1).fit(
        ...     [["ótimo", "produto"], ["péssimo", "atendimento"]], ["positivo", "negativo"]
        ... )  # doctest: +SKIP
        """
        validate_not_empty_collection(X, collection_name="X")
        try:
            import torch  # type: ignore[reportMissingImports]
            from torch import nn  # type: ignore[reportMissingImports]
            from torch.utils.data import (  # type: ignore[reportMissingImports]
                DataLoader,
                TensorDataset,
            )
        except ImportError as exception:
            raise ModelError(
                "A biblioteca 'torch' não está instalada. Instale com `uv add torch` "
                "para treinar o classificador LSTM."
            ) from exception

        seed_everything(self.random_state)
        encoded_sequences, encoded_labels, vocabulary_size, num_classes = (
            self._encode_training_data(X, y)
        )
        embedding_dim = self.embedding_dim
        hidden_dim = self.hidden_dim
        num_layers = self.num_layers
        bidirectional = self.bidirectional
        dropout = self.dropout

        class _LSTMModule(nn.Module):
            """Embedding treinável + (Bi)LSTM + cabeça linear de classificação."""

            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout_layer = nn.Dropout(dropout)
                lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
                self.classifier_head = nn.Linear(lstm_output_dim, num_classes)

            def forward(self, token_ids: "torch.Tensor") -> "torch.Tensor":
                """Classifica um lote de sequências de índices de token.

                Parameters
                ----------
                token_ids : torch.Tensor
                    Lote ``(batch, max_sequence_length)`` de índices de token.

                Returns
                -------
                torch.Tensor
                    Logits ``(batch, num_classes)``.
                """
                embedded = self.embedding(token_ids)
                _, (hidden_state, _) = self.lstm(embedded)
                final_hidden = (
                    torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
                    if bidirectional
                    else hidden_state[-1]
                )
                return self.classifier_head(self.dropout_layer(final_hidden))

        module = _LSTMModule()
        optimizer = torch.optim.Adam(module.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        train_loader, validation_tensor, validation_labels_tensor = self._split_train_validation(
            torch, DataLoader, TensorDataset, encoded_sequences, encoded_labels
        )
        self._run_training_loop(
            module,
            torch,
            optimizer,
            loss_function,
            train_loader,
            validation_tensor,
            validation_labels_tensor,
        )

        self._module = module
        logger.info(
            "LSTM treinada: vocabulário de %d termo(s), %d classe(s).", vocabulary_size, num_classes
        )
        return self

    def predict_proba(self, X: Sequence[Sequence[str]]) -> np.ndarray:  # noqa: N803
        """Estima a distribuição de probabilidade por classe de sentimento.

        Parameters
        ----------
        X : Sequence[Sequence[str]]
            Documentos tokenizados a classificar.

        Returns
        -------
        np.ndarray
            Matriz ``(len(X), n_classes)`` de probabilidades.

        Raises
        ------
        ModelNotFittedError
            Se o modelo ainda não tiver sido treinado via :meth:`fit`.
        """
        if self._module is None or self.vocabulary_ is None:
            raise ModelNotFittedError("LSTMSentimentClassifier")
        import torch  # type: ignore[reportMissingImports]

        encoded_sequences = encode_token_sequences(
            X, self.vocabulary_, max_sequence_length=self.max_sequence_length
        )
        self._module.eval()
        with torch.no_grad():
            logits = self._module(torch.tensor(encoded_sequences, dtype=torch.long))
            return torch.softmax(logits, dim=-1).numpy()

    def predict(self, X: Sequence[Sequence[str]]) -> np.ndarray:  # noqa: N803
        """Prediz o rótulo de sentimento mais provável para cada documento.

        Parameters
        ----------
        X : Sequence[Sequence[str]]
            Documentos tokenizados a classificar.

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


def build_lstm_classifier(**overrides: object) -> LSTMSentimentClassifier:
    """Fábrica do classificador LSTM/BiLSTM.

    Parameters
    ----------
    **overrides : object
        Hiperparâmetros que sobrescrevem os padrões do construtor de
        :class:`LSTMSentimentClassifier` (espelhando
        ``configs/model_params.yaml -> deep_learning.recurrent``).

    Returns
    -------
    LSTMSentimentClassifier
        Classificador não treinado, pronto para ``fit``.

    Examples
    --------
    >>> build_lstm_classifier(hidden_dim=64).hidden_dim
    64
    """
    logger.info("Construindo LSTMSentimentClassifier (overrides=%s).", overrides)
    return LSTMSentimentClassifier(**overrides)  # type: ignore[arg-type]
