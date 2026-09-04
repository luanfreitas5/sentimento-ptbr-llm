"""Sparse Autoencoder Top-K (com perda Matryoshka opcional) para o HypotheSAEs.

Núcleo não-supervisionado do método: treina um autoencoder esparso sobre
embeddings de texto (ver ``embedding.py``), cujos neurônios ativos (top-K
por exemplo) tendem a corresponder a conceitos interpretáveis, que depois
são nomeados por um LLM (``interpret_neurons.py``) e usados como hipóteses
sobre o corpus de tweets.

Detalhes de implementação:

1. Perda Matryoshka: a perda de reconstrução é calculada em prefixos
   crescentes de neurônios e depois promediada, produzindo neurônios com
   granularidades diferentes (poucos neurônios grosseiros, muitos
   neurônios finos).
2. Reconstrução auxiliar (aux-K): revive neurônios "mortos" (que não
   ativam há muitos passos), usando-os para prever o resíduo de
   reconstrução dos neurônios ativos.
3. Reconstrução multi-K: uma reconstrução secundária, menos esparsa,
   opcional, com peso menor na perda total.
4. Batch Top-K (opcional): seleciona as top-(K·tamanho_do_lote) ativações
   em todo o lote durante o treino, mantendo um limiar aprendido usado na
   inferência para preservar a esparsidade esperada.

Esta implementação segue de perto:

- Bussmann et al. (2025), ``matryoshka_sae`` (perda Matryoshka)
- O'Neill et al. (2024), ``saerch`` (Top-K com revivificação de neurônios
  mortos)

``torch`` não está nas dependências base do projeto (dependência pesada e
opcional). Como este módulo inteiro gira em torno de um ``torch.nn.Module``,
o import é feito no topo do arquivo, mas guardado por um erro de projeto
claro (:class:`~exceptions.model.ModelError`) caso a biblioteca esteja
ausente, em vez de propagar um ``ModuleNotFoundError`` cru. Instale com
``uv add torch`` antes de treinar ou carregar um SAE.
"""

import logging
from pathlib import Path

import numpy as np

from exceptions.model import ModelError
from utils.validation import validate_file_exists

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.nn import functional
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
except ImportError as _import_error:  # pragma: no cover - guarda defensiva
    raise ModelError(
        "A biblioteca 'torch' não está instalada. Instale com `uv add torch` "
        "para treinar ou carregar um Sparse Autoencoder do HypotheSAEs."
    ) from _import_error


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder Top-K, com suporte a perda Matryoshka e Batch Top-K.

    Parameters
    ----------
    input_dim : int
        Dimensionalidade dos embeddings de entrada.
    m_total_neurons : int
        Número total de neurônios (features) do SAE.
    k_active_neurons : int
        Número de neurônios ativos (top-K) selecionados por exemplo.
    aux_k : int | None, optional
        Limite superior de neurônios "mortos" considerados na reconstrução
        auxiliar. Por padrão, ``min(2 * k_active_neurons, m_total_neurons)``.
        O coeficiente padrão da perda auxiliar é 1/32 (ver
        :meth:`compute_loss`).
    multi_k : int | None, optional
        Número de neurônios usados na reconstrução multi-K (menos esparsa).
        Por padrão ``None`` (sem reconstrução multi-K); um valor inicial
        recomendado é ``4 * k_active_neurons``.
    dead_neuron_threshold_steps : int, optional
        Número de passos sem ativação após os quais um neurônio é
        considerado "morto", by default 256.
    prefix_lengths : list[int] | None, optional
        Se informado (ex.: ``[16, 64]``), ativa a perda *Matryoshka*: o
        primeiro prefixo tem 16 neurônios, o segundo 64, etc. Se ``None``,
        todos os ``m_total_neurons`` neurônios são tratados igualmente.
    use_batch_topk : bool, optional
        Se ``True``, usa esparsidade Top-K em lote (com limiar aprendido
        para inferência), by default False.
    device : str, optional
        Dispositivo PyTorch, by default "cuda" se disponível, senão "cpu".
    """

    def __init__(
        self,
        input_dim: int,
        m_total_neurons: int,
        k_active_neurons: int,
        *,
        aux_k: int | None = None,
        multi_k: int | None = None,
        dead_neuron_threshold_steps: int = 256,
        prefix_lengths: list[int] | None = None,
        use_batch_topk: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.m_total_neurons = m_total_neurons
        self.k_active_neurons = k_active_neurons
        self.use_batch_topk = use_batch_topk

        self.aux_k = min(2 * k_active_neurons, m_total_neurons) if aux_k is None else aux_k
        self.multi_k = multi_k
        self.dead_neuron_threshold_steps = dead_neuron_threshold_steps

        self.prefix_lengths = prefix_lengths
        if self.prefix_lengths is not None:
            if self.prefix_lengths[-1] != m_total_neurons:
                raise ValueError("O último prefix_length deve ser igual a m_total_neurons")
            if not all(
                x > y
                for x, y in zip(self.prefix_lengths[1:], self.prefix_lengths[:-1], strict=True)
            ):
                raise ValueError("Cada prefix_length deve ser maior que o anterior")

        self.encoder = nn.Linear(input_dim, m_total_neurons, bias=False)
        self.decoder = nn.Linear(m_total_neurons, input_dim, bias=False)

        self.input_bias = nn.Parameter(torch.zeros(input_dim))
        self.neuron_bias = nn.Parameter(torch.zeros(m_total_neurons))

        self.steps_since_activation = torch.zeros(m_total_neurons, dtype=torch.long, device=device)

        # Limiar de Batch Top-K (usado somente quando use_batch_topk=True).
        self.register_buffer("threshold", torch.tensor(0.0))

        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """Executa o forward pass (agnóstico à configuração Matryoshka).

        Parameters
        ----------
        x : torch.Tensor
            Lote de embeddings de entrada, formato ``(n, input_dim)``.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor | None]]
            Par (reconstrução, dicionário com ativações e informações
            auxiliares usadas por :meth:`compute_loss`).
        """
        x = x - self.input_bias
        pre_activation = self.encoder(x) + self.neuron_bias

        topk_indices = topk_values = None
        if not self.use_batch_topk:
            local_values, local_indices = torch.topk(pre_activation, self.k_active_neurons, dim=-1)
            local_values = functional.relu(local_values)
            activations = torch.zeros_like(pre_activation)
            activations.scatter_(-1, local_indices, local_values)
            topk_indices, topk_values = local_indices, local_values
        else:
            relu_activations = functional.relu(pre_activation)
            if self.training:
                batch_size = relu_activations.shape[0]
                flat_activations = relu_activations.flatten()
                k_total = min(self.k_active_neurons * batch_size, flat_activations.numel())
                flat_values, flat_indices = torch.topk(flat_activations, k_total, dim=-1)
                flat_scattered = torch.zeros_like(flat_activations)
                flat_scattered.scatter_(0, flat_indices, flat_values)
                activations = flat_scattered.view_as(relu_activations)
                self._update_threshold_(activations)
            else:
                activations = torch.where(
                    relu_activations > self.threshold,
                    relu_activations,
                    torch.zeros_like(relu_activations),
                )

        if self.multi_k is not None:
            multik_values, multik_indices = torch.topk(pre_activation, self.multi_k, dim=-1)
            multik_values = functional.relu(multik_values)
            multik_activations = torch.zeros_like(pre_activation)
            multik_activations.scatter_(-1, multik_indices, multik_values)
            multik_reconstruction = self.decoder(multik_activations) + self.input_bias
        else:
            multik_reconstruction = None

        self.steps_since_activation += 1
        fired = (activations.sum(dim=0) > 0).nonzero(as_tuple=False).squeeze(-1)
        if fired.numel() > 0:
            self.steps_since_activation.index_fill_(0, fired, 0)

        reconstruction = self.decoder(activations) + self.input_bias

        aux_indices = aux_values = None
        if self.aux_k is not None:
            dead_mask = (self.steps_since_activation > self.dead_neuron_threshold_steps).float()
            dead_pre_activation = pre_activation * dead_mask
            aux_values, aux_indices = torch.topk(dead_pre_activation, self.aux_k, dim=-1)
            aux_values = functional.relu(aux_values)

        info: dict[str, torch.Tensor | None] = {
            "activations": activations,
            "topk_indices": topk_indices,
            "topk_values": topk_values,
            "multik_reconstruction": multik_reconstruction,
            "aux_indices": aux_indices,
            "aux_values": aux_values,
        }
        return reconstruction, info

    @staticmethod
    def _normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula o MSE normalizado pelo MSE da predição baseline (média por coluna)."""
        mse = functional.mse_loss(prediction, target)
        baseline_mse = functional.mse_loss(
            target.mean(dim=0, keepdim=True).expand_as(target), target
        )
        return mse / baseline_mse

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        info: dict[str, torch.Tensor | None],
        aux_coef: float,
        multi_coef: float,
    ) -> torch.Tensor:
        """Calcula a perda total (L2 Matryoshka + multi-K opcional + aux-K opcional).

        Se ``len(prefix_lengths) == 1`` (ou ``prefix_lengths is None``), não
        há aninhamento Matryoshka. Caso contrário, calcula a média do L2 de
        cada prefixo de reconstrução (Bussmann et al., 2025). Os termos
        multi-K e aux-K seguem O'Neill et al. (2024).

        Parameters
        ----------
        x : torch.Tensor
            Lote de embeddings de entrada.
        reconstruction : torch.Tensor
            Reconstrução completa, retornada por :meth:`forward`.
        info : dict[str, torch.Tensor | None]
            Dicionário auxiliar retornado por :meth:`forward`.
        aux_coef : float
            Coeficiente da perda auxiliar (aux-K).
        multi_coef : float
            Coeficiente da perda multi-K.

        Returns
        -------
        torch.Tensor
            Escalar de perda total, pronto para ``.backward()``.
        """
        activations = info["activations"]
        assert activations is not None

        if self.prefix_lengths is None or len(self.prefix_lengths) == 1:
            main_l2 = self._normalized_mse(reconstruction, x)
        else:
            decoder_weight = self.decoder.weight  # (input_dim, m_total_neurons)
            l2_terms = [
                self._normalized_mse(
                    activations[:, :end] @ decoder_weight[:, :end].t() + self.input_bias, x
                )
                for end in self.prefix_lengths
            ]
            main_l2 = torch.stack(l2_terms).mean()

        if multi_coef != 0 and info["multik_reconstruction"] is not None:
            main_l2 = main_l2 + multi_coef * self._normalized_mse(info["multik_reconstruction"], x)

        if self.aux_k is not None and info["aux_indices"] is not None:
            residual = x - reconstruction.detach()
            aux_activations = torch.zeros_like(activations)
            aux_activations.scatter_(-1, info["aux_indices"], info["aux_values"])
            residual_reconstruction = self.decoder(aux_activations)
            aux_loss = self._normalized_mse(residual_reconstruction, residual)
            return main_l2 + aux_coef * aux_loss

        return main_l2

    def normalize_decoder_(self) -> None:
        """Normaliza cada coluna do decoder para norma unitária (in-place)."""
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))

    def adjust_decoder_gradient_(self) -> None:
        """Remove do gradiente do decoder a componente paralela aos pesos (in-place).

        Mantém o decoder com colunas de norma unitária ao longo do
        treinamento, projetando o gradiente ortogonalmente aos pesos atuais
        antes do passo do otimizador.
        """
        if self.decoder.weight.grad is not None:
            with torch.no_grad():
                projection = (self.decoder.weight * self.decoder.weight.grad).sum(
                    dim=0, keepdim=True
                )
                self.decoder.weight.grad.sub_(projection * self.decoder.weight)

    def initialize_weights_(self, data_sample: torch.Tensor) -> None:
        """Inicializa os pesos do SAE a partir de uma amostra de dados (in-place)."""
        self.input_bias.data = torch.median(data_sample, dim=0).values
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder_()
        self.encoder.weight.data = self.decoder.weight.t().clone()
        nn.init.zeros_(self.neuron_bias)

    @torch.no_grad()
    def _update_threshold_(self, activations: torch.Tensor, learning_rate: float = 1e-2) -> None:
        """Atualiza (EMA) o limiar em direção à menor ativação positiva do lote."""
        positive_mask = activations > 0
        if positive_mask.any():
            min_positive = activations[positive_mask].min()
            self.threshold.mul_(1 - learning_rate).add_(learning_rate * min_positive)

    def save(self, save_path: Path) -> Path:
        """Salva a configuração e os pesos do modelo em um checkpoint ``.pt``.

        Parameters
        ----------
        save_path : Path
            Caminho de destino do checkpoint.

        Returns
        -------
        Path
            O mesmo ``save_path``, para encadeamento.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "input_dim": self.input_dim,
            "m_total_neurons": self.m_total_neurons,
            "k_active_neurons": self.k_active_neurons,
            "aux_k": self.aux_k,
            "multi_k": self.multi_k,
            "dead_neuron_threshold_steps": self.dead_neuron_threshold_steps,
            "prefix_lengths": self.prefix_lengths,
            "use_batch_topk": self.use_batch_topk,
        }
        torch.save({"config": config, "state_dict": self.state_dict()}, save_path)
        logger.info("Modelo salvo em '%s'.", save_path)
        return save_path

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        aux_coef: float,
        multi_coef: float,
        clip_grad: float | None,
    ) -> float:
        """Executa uma época de treino sobre ``train_loader`` e retorna a perda média."""
        self.train()
        train_losses = []

        for (raw_batch_x,) in train_loader:
            batch_x = raw_batch_x.to(self.device)
            reconstruction, info = self(batch_x)
            loss = self.compute_loss(batch_x, reconstruction, info, aux_coef, multi_coef)

            optimizer.zero_grad()
            loss.backward()
            self.adjust_decoder_gradient_()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

            optimizer.step()
            self.normalize_decoder_()

            train_losses.append(loss.item())

        return float(np.mean(train_losses))

    def _evaluate_validation_loss(
        self, val_loader: DataLoader, aux_coef: float, multi_coef: float
    ) -> float:
        """Calcula a perda média de validação sobre ``val_loader``, sem atualizar pesos."""
        self.eval()
        val_losses = []
        with torch.no_grad():
            for (raw_batch_x,) in val_loader:
                batch_x = raw_batch_x.to(self.device)
                reconstruction, info = self(batch_x)
                val_losses.append(
                    self.compute_loss(batch_x, reconstruction, info, aux_coef, multi_coef).item()
                )

        return float(np.mean(val_losses))

    def fit(
        self,
        x_train: torch.Tensor,
        x_val: torch.Tensor | None = None,
        save_dir: Path | None = None,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        n_epochs: int = 200,
        aux_coef: float = 1 / 32,
        multi_coef: float = 0.0,
        patience: int = 5,
        show_progress: bool = True,
        clip_grad: float | None = 1.0,
    ) -> dict[str, list[float]]:
        """Treina o Sparse Autoencoder sobre os dados de entrada.

        Parameters
        ----------
        x_train : torch.Tensor
            Embeddings de treino, formato ``(n, input_dim)``.
        x_val : torch.Tensor | None, optional
            Embeddings de validação, usados para early stopping, by default
            None.
        save_dir : Path | None, optional
            Diretório onde salvar o checkpoint final (nome gerado por
            :func:`build_sae_checkpoint_name`), by default None.
        batch_size : int, optional
            Tamanho do lote de treino, by default 512.
        learning_rate : float, optional
            Taxa de aprendizado do otimizador Adam, by default 5e-4.
        n_epochs : int, optional
            Número máximo de épocas, by default 200.
        aux_coef : float, optional
            Coeficiente da perda auxiliar (aux-K), by default 1/32.
        multi_coef : float, optional
            Coeficiente da perda multi-K, by default 0.0.
        patience : int, optional
            Número de épocas sem melhora na perda de validação antes de
            interromper o treinamento, by default 5.
        show_progress : bool, optional
            Se exibe barra de progresso, by default True.
        clip_grad : float | None, optional
            Valor máximo de norma de gradiente (clipping), by default 1.0.

        Returns
        -------
        dict[str, list[float]]
            Histórico de treino: ``train_loss``, ``val_loss`` e
            ``dead_neuron_ratio`` por época.
        """
        train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(TensorDataset(x_val), batch_size=batch_size) if x_val is not None else None
        )

        self.initialize_weights_(x_train.to(self.device))

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        patience_counter = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "dead_neuron_ratio": [],
        }
        avg_val_loss = None

        iterator = tqdm(range(n_epochs), disable=not show_progress)
        for epoch in iterator:
            avg_train_loss = self._train_one_epoch(
                train_loader, optimizer, aux_coef, multi_coef, clip_grad
            )
            history["train_loss"].append(avg_train_loss)

            dead_ratio = (
                (self.steps_since_activation > self.dead_neuron_threshold_steps)
                .float()
                .mean()
                .item()
            )
            history["dead_neuron_ratio"].append(dead_ratio)

            if val_loader is not None:
                avg_val_loss = self._evaluate_validation_loss(val_loader, aux_coef, multi_coef)
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping ativado após %d época(s).", epoch + 1)
                        break

            if show_progress:
                postfix = {
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_loss": f"{avg_val_loss:.4f}" if val_loader else "N/A",
                    "dead_ratio": f"{dead_ratio:.3f}",
                }
                if self.use_batch_topk:
                    postfix["threshold"] = f"{self.threshold.item():.2e}"
                iterator.set_postfix(postfix)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_name = build_sae_checkpoint_name(
                self.m_total_neurons, self.k_active_neurons, self.prefix_lengths
            )
            self.save(save_dir / checkpoint_name)

        return history

    def compute_activations(
        self,
        inputs: list | np.ndarray | torch.Tensor,
        batch_size: int = 16384,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Calcula as ativações esparsas do SAE para os dados de entrada, em lotes.

        O processamento em lotes evita estourar a memória da GPU (CUDA OOM)
        em conjuntos de dados grandes.

        Parameters
        ----------
        inputs : list | np.ndarray | torch.Tensor
            Dados de entrada (embeddings).
        batch_size : int, optional
            Número de amostras por lote, by default 16384.
        show_progress : bool, optional
            Se exibe barra de progresso, by default True.

        Returns
        -------
        np.ndarray
            Matriz ``(n, m_total_neurons)`` de ativações esparsas.

        Raises
        ------
        TypeError
            Se ``inputs`` não for uma lista, array numpy ou tensor torch.
        """
        self.eval()

        if isinstance(inputs, list):
            tensor_inputs = torch.tensor(inputs, dtype=torch.float)
        elif isinstance(inputs, np.ndarray):
            tensor_inputs = torch.from_numpy(inputs).float()
        elif isinstance(inputs, torch.Tensor):
            tensor_inputs = inputs
        else:
            raise TypeError("inputs deve ser uma lista, array numpy ou tensor torch")
        if tensor_inputs.dtype != torch.float:
            tensor_inputs = tensor_inputs.float()

        num_samples = tensor_inputs.shape[0]
        all_activations = []
        with torch.no_grad():
            index_range: object = range(0, num_samples, batch_size)
            if show_progress:
                index_range = tqdm(
                    index_range, desc=f"Calculando ativações (tamanho do lote={batch_size})"
                )

            for i in index_range:
                batch = tensor_inputs[i : i + batch_size].to(self.device)
                _, info = self(batch)
                all_activations.append(info["activations"].cpu())

        return torch.cat(all_activations, dim=0).numpy()


def build_sae_checkpoint_name(
    m_total_neurons: int, k_active_neurons: int, prefix_lengths: list[int] | None = None
) -> str:
    """Monta o nome de arquivo padronizado de um checkpoint de SAE.

    Parameters
    ----------
    m_total_neurons : int
        Número total de neurônios do SAE.
    k_active_neurons : int
        Número de neurônios ativos (top-K).
    prefix_lengths : list[int] | None, optional
        Prefixos usados na perda Matryoshka, se houver, by default None.

    Returns
    -------
    str
        Nome de arquivo ``.pt`` determinístico para a configuração.

    Examples
    --------
    >>> build_sae_checkpoint_name(256, 8)
    'SAE_M=256_K=8.pt'
    >>> build_sae_checkpoint_name(256, 8, [32, 256])
    'SAE_matryoshka_M=256_K=8_prefixes=32-256.pt'
    """
    if prefix_lengths is None:
        return f"SAE_M={m_total_neurons}_K={k_active_neurons}.pt"
    prefix_str = "-".join(str(length) for length in prefix_lengths)
    return f"SAE_matryoshka_M={m_total_neurons}_K={k_active_neurons}_prefixes={prefix_str}.pt"


def load_model(
    path: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SparseAutoencoder:
    """Carrega um Sparse Autoencoder previamente salvo com :meth:`SparseAutoencoder.save`.

    Parameters
    ----------
    path : Path
        Caminho do checkpoint ``.pt``.
    device : str, optional
        Dispositivo PyTorch de destino, by default "cuda" se disponível,
        senão "cpu".

    Returns
    -------
    SparseAutoencoder
        Modelo com os pesos carregados, no modo de avaliação.

    Raises
    ------
    DataNotFoundError
        Se ``path`` não existir.

    Examples
    --------
    >>> load_model(Path("checkpoints/inexistente.pt"))
    Traceback (most recent call last):
        ...
    exceptions.data.DataNotFoundError: ...
    """
    validate_file_exists(path)
    # weights_only=True: o checkpoint contém apenas tensores e um dict de
    # configuração com tipos primitivos (int, bool, list[int], None),
    # todos cobertos pelo unpickler restrito do PyTorch.
    checkpoint = torch.load(path, weights_only=True)
    model = SparseAutoencoder(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info("Modelo carregado de '%s' no dispositivo '%s'.", path, model.device)
    return model
