"""Adaptador do autoencoder de redução de dimensionalidade à interface de modelo do projeto.

Implementa a Fase 9 (Seções 4.4-4.5 do documento mestre). Este módulo não
reimplementa o treinamento/codificação do autoencoder - já implementado em
``src/features/reduction.py`` (Fase 8) - apenas o expõe como um
transformador ``fit``/``transform`` com hiperparâmetros padrão alinhados a
``configs/model_params.yaml -> autoencoder``, para uso uniforme em
``src/models/factory.py`` e nos pipelines de treino clássico (redução de
embeddings contextuais antes dos classificadores clássicos).
"""

import logging
from collections.abc import Sequence

import numpy as np

from exceptions.model import ModelNotFittedError
from features.reduction import (
    AutoencoderArtifacts,
    compute_reconstruction_error,
    encode_with_autoencoder,
    train_autoencoder,
)

logger = logging.getLogger(__name__)


class AutoencoderFeatureReducer:
    """Transformador ``fit``/``transform`` que reduz embeddings via autoencoder.

    Envolve :func:`features.reduction.train_autoencoder` e
    :func:`features.reduction.encode_with_autoencoder` em uma interface no
    estilo scikit-learn, para uso intercambiável com os demais modelos de
    ``src/models/`` a partir de ``src/models/factory.py``.

    Parameters
    ----------
    input_dim : int, optional
        Dimensão de entrada dos embeddings, by default 768 (BERTimbau base).
    latent_dim : int, optional
        Dimensão do espaço latente, by default 128.
    hidden_layers : Sequence[int], optional
        Tamanho das camadas ocultas do encoder, by default (512, 256).
    activation : str, optional
        Função de ativação, by default "relu".
    dropout : float, optional
        Taxa de dropout, by default 0.2.
    learning_rate : float, optional
        Taxa de aprendizado do otimizador Adam, by default 0.001.
    batch_size : int, optional
        Tamanho do lote de treino, by default 64.
    epochs : int, optional
        Número máximo de épocas, by default 50.
    early_stopping_patience : int, optional
        Épocas sem melhora na perda de validação antes de interromper o
        treino, by default 5.
    random_state : int, optional
        Semente aleatória, by default 42.
    """

    def __init__(
        self,
        *,
        input_dim: int = 768,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 256),
        activation: str = "relu",
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self._artifacts: AutoencoderArtifacts | None = None

    def fit(self, X: np.ndarray, y: object = None) -> "AutoencoderFeatureReducer":  # noqa: ARG002
        """Treina o autoencoder sobre a matriz de embeddings de entrada.

        Parameters
        ----------
        X : np.ndarray
            Matriz ``(n_amostras, input_dim)`` de embeddings de treino.
        y : object, optional
            Ignorado; mantido apenas pela paridade com a API scikit-learn
            (``fit(X, y)``), by default None.

        Returns
        -------
        AutoencoderFeatureReducer
            A própria instância, treinada.

        Raises
        ------
        EmptyDatasetError
            Se ``X`` estiver vazio.
        ModelError
            Se ``torch`` não estiver instalado.
        """
        self._artifacts = train_autoencoder(
            X,
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,
            early_stopping_patience=self.early_stopping_patience,
            random_state=self.random_state,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projeta embeddings no espaço latente aprendido.

        Parameters
        ----------
        X : np.ndarray
            Matriz ``(n_amostras, input_dim)`` de embeddings a projetar.

        Returns
        -------
        np.ndarray
            Matriz ``(n_amostras, latent_dim)`` de representações latentes.

        Raises
        ------
        ModelNotFittedError
            Se o autoencoder ainda não tiver sido treinado via :meth:`fit`.
        EmptyDatasetError
            Se ``X`` estiver vazio.
        """
        if self._artifacts is None:
            raise ModelNotFittedError("AutoencoderFeatureReducer")
        return encode_with_autoencoder(X, self._artifacts)

    def fit_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """Treina o autoencoder e projeta ``X`` no espaço latente em seguida.

        Parameters
        ----------
        X : np.ndarray
            Matriz ``(n_amostras, input_dim)`` de embeddings de treino.
        y : object, optional
            Ignorado, by default None.

        Returns
        -------
        np.ndarray
            Matriz ``(n_amostras, latent_dim)`` de representações latentes.
        """
        return self.fit(X, y).transform(X)

    def score_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calcula o erro de reconstrução por amostra, como sinal diagnóstico.

        Parameters
        ----------
        X : np.ndarray
            Matriz ``(n_amostras, input_dim)`` de embeddings a avaliar.

        Returns
        -------
        np.ndarray
            Vetor ``(n_amostras,)`` com o erro de reconstrução de cada
            amostra.

        Raises
        ------
        ModelNotFittedError
            Se o autoencoder ainda não tiver sido treinado via :meth:`fit`.
        EmptyDatasetError
            Se ``X`` estiver vazio.
        """
        if self._artifacts is None:
            raise ModelNotFittedError("AutoencoderFeatureReducer")
        return compute_reconstruction_error(X, self._artifacts)


def build_autoencoder_reducer(**overrides: object) -> AutoencoderFeatureReducer:
    """Fábrica do redutor de dimensionalidade via autoencoder.

    Parameters
    ----------
    **overrides : object
        Hiperparâmetros que sobrescrevem os padrões do construtor de
        :class:`AutoencoderFeatureReducer` (espelhando
        ``configs/model_params.yaml -> autoencoder``).

    Returns
    -------
    AutoencoderFeatureReducer
        Transformador não treinado, pronto para ``fit``.

    Examples
    --------
    >>> build_autoencoder_reducer(latent_dim=64).latent_dim
    64
    """
    logger.info("Construindo AutoencoderFeatureReducer (overrides=%s).", overrides)
    return AutoencoderFeatureReducer(**overrides)  # type: ignore[arg-type]
