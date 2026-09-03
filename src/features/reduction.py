"""Redução de dimensionalidade de embeddings via autoencoder (PyTorch).

Implementa a Seção 4.4 do documento mestre, no duplo papel do autoencoder:
(1) comprimir embeddings contextuais de alta dimensão (768, BERTimbau base)
em um espaço latente menor (128, ver ``configs/model_params.yaml ->
autoencoder``) antes de alimentar os classificadores clássicos, e (2) usar o
erro de reconstrução por amostra como sinal diagnóstico não supervisionado
de tweets atípicos/ambíguos (complementar ao HypotheSAEs, ``src/labeling/``).

``torch`` é uma dependência pesada e opcional, ainda não instalada no
projeto: o import ocorre de forma tardia, dentro das funções deste módulo,
para que o módulo permaneça importável sem ela.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from exceptions.model import ModelError
from utils.seed import seed_everything
from utils.validation import validate_not_empty_collection

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


@dataclass
class AutoencoderArtifacts:
    """Artefatos produzidos pelo treinamento do autoencoder.

    Parameters
    ----------
    module : Any
        Módulo PyTorch treinado (``torch.nn.Module``), tipado como ``Any``
        para que este módulo permaneça importável sem ``torch`` instalado.
    input_dim : int
        Dimensão de entrada (e de reconstrução) esperada pelo modelo.
    latent_dim : int
        Dimensão do espaço latente produzido por
        :func:`encode_with_autoencoder`.
    training_loss_history : list[float]
        Perda de reconstrução (MSE) média por época de treinamento.
    """

    module: Any
    input_dim: int
    latent_dim: int
    training_loss_history: list[float]


def _build_autoencoder_module(
    *,
    input_dim: int,
    latent_dim: int,
    hidden_layers: Sequence[int],
    activation: str,
    dropout: float,
) -> "torch.nn.Module":
    """Constrói a arquitetura encoder-decoder simétrica do autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensão de entrada e de reconstrução.
    latent_dim : int
        Dimensão do gargalo (espaço latente).
    hidden_layers : Sequence[int]
        Tamanho de cada camada oculta do encoder; o decoder usa as mesmas
        camadas em ordem inversa.
    activation : str
        Nome da função de ativação (``"relu"``, ``"tanh"`` ou ``"gelu"``).
    dropout : float
        Taxa de dropout aplicada após cada camada oculta.

    Returns
    -------
    torch.nn.Module
        Módulo PyTorch não treinado, pronto para :func:`train_autoencoder`.

    Raises
    ------
    ValueError
        Se ``activation`` não for uma das funções suportadas.
    """
    from torch import nn  # type: ignore[reportMissingImports]

    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }
    if activation not in activations:
        raise ValueError(
            f"Ativação '{activation}' não suportada. Valores permitidos: {list(activations)}"
        )
    activation_layer = activations[activation]

    def _build_multilayer_perceptron(layer_sizes: Sequence[int]) -> nn.Sequential:
        """Constrói um perceptron multicamadas a partir dos tamanhos de camada informados.

        Parameters
        ----------
        layer_sizes : Sequence[int]
            Tamanho de cada camada, da entrada à saída.

        Returns
        -------
        nn.Sequential
            Sequência de camadas lineares, com ativação e dropout entre
            elas (exceto após a última camada).
        """
        layers: list[nn.Module] = []
        for in_features, out_features in pairwise(layer_sizes):
            layers.append(nn.Linear(in_features, out_features))
            if out_features != layer_sizes[-1]:
                layers.extend([activation_layer()])
                layers.extend([nn.Dropout(dropout)])
        return nn.Sequential(*layers)

    encoder_layer_sizes = [input_dim, *hidden_layers, latent_dim]
    decoder_layer_sizes = [latent_dim, *reversed(hidden_layers), input_dim]

    class _AutoencoderModule(nn.Module):
        """Autoencoder simétrico encoder-decoder com gargalo em ``latent_dim``."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = _build_multilayer_perceptron(encoder_layer_sizes)
            self.decoder = _build_multilayer_perceptron(decoder_layer_sizes)

        def forward(self, batch: "torch.Tensor") -> "torch.Tensor":
            """Reconstrói o lote de entrada, passando pelo espaço latente.

            Parameters
            ----------
            batch : torch.Tensor
                Lote de embeddings de entrada, formato ``(n, input_dim)``.

            Returns
            -------
            torch.Tensor
                Reconstrução do lote, mesmo formato de ``batch``.
            """
            return self.decoder(self.encoder(batch))

    return _AutoencoderModule()


def train_autoencoder(
    training_embeddings: np.ndarray,
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
) -> AutoencoderArtifacts:
    """Treina um autoencoder para reduzir a dimensionalidade de embeddings.

    Reserva 10% das amostras para validação (parada antecipada monitorando
    a perda de reconstrução de validação); todos os hiperparâmetros
    espelham ``configs/model_params.yaml -> autoencoder``.

    Parameters
    ----------
    training_embeddings : np.ndarray
        Matriz ``(n_amostras, input_dim)`` de embeddings de treino (ex.:
        saída de
        ``src.features.contextual_embeddings.extract_contextual_embeddings``).
    input_dim : int, optional
        Dimensão de entrada, by default 768.
    latent_dim : int, optional
        Dimensão do espaço latente, by default 128.
    hidden_layers : Sequence[int], optional
        Tamanho das camadas ocultas do encoder (decoder em ordem inversa),
        by default (512, 256).
    activation : str, optional
        Repassado à construção da arquitetura, by default "relu".
    dropout : float, optional
        Taxa de dropout, by default 0.2.
    learning_rate : float, optional
        Taxa de aprendizado do otimizador Adam, by default 0.001.
    batch_size : int, optional
        Tamanho do lote de treinamento, by default 64.
    epochs : int, optional
        Número máximo de épocas, by default 50.
    early_stopping_patience : int, optional
        Número de épocas sem melhora na perda de validação antes de
        interromper o treinamento, by default 5.
    random_state : int, optional
        Semente aleatória, aplicada via :func:`utils.seed.seed_everything`
        antes do treinamento, by default 42.

    Returns
    -------
    AutoencoderArtifacts
        Artefatos do modelo treinado.

    Raises
    ------
    EmptyDatasetError
        Se ``training_embeddings`` estiver vazio.
    ModelError
        Se ``torch`` não estiver instalado.

    Examples
    --------
    >>> embeddings = np.random.default_rng(42).normal(size=(100, 768))
    >>> train_autoencoder(embeddings, epochs=1)  # doctest: +SKIP
    """
    validate_not_empty_collection(training_embeddings, collection_name="training_embeddings")
    try:
        import torch  # type: ignore[reportMissingImports]
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore[reportMissingImports]
    except ImportError as exception:
        raise ModelError(
            "A biblioteca 'torch' não está instalada. Instale com `uv add torch` "
            "para treinar o autoencoder de redução de dimensionalidade."
        ) from exception

    seed_everything(random_state)
    module = _build_autoencoder_module(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    validation_size = max(1, int(0.1 * len(training_embeddings)))
    permutation = np.random.default_rng(random_state).permutation(len(training_embeddings))
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    train_tensor = torch.tensor(training_embeddings[train_indices], dtype=torch.float32)
    validation_tensor = torch.tensor(training_embeddings[validation_indices], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    training_loss_history: list[float] = []

    for epoch in range(epochs):
        module.train()
        epoch_losses: list[float] = []
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstruction = module(batch)
            loss = loss_function(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        training_loss_history.append(float(np.mean(epoch_losses)))

        module.eval()
        with torch.no_grad():
            validation_loss = loss_function(module(validation_tensor), validation_tensor).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Parada antecipada na época %d (sem melhora por %d época(s)).",
                epoch,
                early_stopping_patience,
            )
            break

    logger.info(
        "Autoencoder treinado: %d época(s), perda de validação final %.4f.",
        len(training_loss_history),
        best_validation_loss,
    )
    return AutoencoderArtifacts(
        module=module,
        input_dim=input_dim,
        latent_dim=latent_dim,
        training_loss_history=training_loss_history,
    )


def encode_with_autoencoder(embeddings: np.ndarray, artifacts: AutoencoderArtifacts) -> np.ndarray:
    """Projeta embeddings no espaço latente aprendido pelo autoencoder.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz ``(n_amostras, artifacts.input_dim)`` de embeddings a
        projetar.
    artifacts : AutoencoderArtifacts
        Artefatos de um autoencoder treinado, via :func:`train_autoencoder`.

    Returns
    -------
    np.ndarray
        Matriz ``(n_amostras, artifacts.latent_dim)`` de representações
        latentes, usável como entrada reduzida para os classificadores
        clássicos.

    Raises
    ------
    EmptyDatasetError
        Se ``embeddings`` estiver vazio.

    Examples
    --------
    >>> encode_with_autoencoder(np.zeros((10, 768)), artifacts)  # doctest: +SKIP
    """
    validate_not_empty_collection(embeddings, collection_name="embeddings")
    import torch  # type: ignore[reportMissingImports]

    artifacts.module.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(embeddings, dtype=torch.float32)
        return artifacts.module.encoder(input_tensor).numpy()


def compute_reconstruction_error(
    embeddings: np.ndarray, artifacts: AutoencoderArtifacts
) -> np.ndarray:
    """Calcula o erro de reconstrução (MSE) por amostra, como sinal diagnóstico.

    Amostras com erro de reconstrução muito acima da média são candidatas a
    tweets atípicos/ambíguos, complementando o diagnóstico não supervisionado
    do HypotheSAEs (ver Seção 4.4 do documento mestre).

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz ``(n_amostras, artifacts.input_dim)`` de embeddings a
        avaliar.
    artifacts : AutoencoderArtifacts
        Artefatos de um autoencoder treinado, via :func:`train_autoencoder`.

    Returns
    -------
    np.ndarray
        Vetor ``(n_amostras,)`` com o erro quadrático médio de reconstrução
        de cada amostra.

    Raises
    ------
    EmptyDatasetError
        Se ``embeddings`` estiver vazio.

    Examples
    --------
    >>> compute_reconstruction_error(np.zeros((10, 768)), artifacts)  # doctest: +SKIP
    """
    validate_not_empty_collection(embeddings, collection_name="embeddings")
    import torch  # type: ignore[reportMissingImports]

    artifacts.module.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(embeddings, dtype=torch.float32)
        reconstruction = artifacts.module(input_tensor)
        return ((reconstruction - input_tensor) ** 2).mean(dim=1).numpy()
