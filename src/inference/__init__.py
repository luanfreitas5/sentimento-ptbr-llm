"""Camada comum de inferência para os quatro paradigmas de modelo do projeto.

Implementa a Fase 12 do plano de elaboração (``PLANO-ELABORACAO.md``): uma
interface única de predição (:class:`inference.predictor.Predictor`) sobre
qualquer modelo que satisfaça :class:`models.base.SentimentClassifier` (ML
clássico, DL, Transformer ou LLM), com padronização determinística da saída
(``src/inference/postprocessing.py``) e três modos de uso — em lote
(``src/inference/batch.py``), ponto a ponto
(``src/inference/online.py``) e em lote concorrente para LLMs
(``src/inference/llm_batch.py``).

Modules
-------
postprocessing
    Padronização e parsing determinístico das saídas de inferência.
predictor
    Interface única de inferência por modelo.
batch
    Inferência em lote (classificadores treinados).
online
    Inferência ponto a ponto (uso interativo/API).
llm_batch
    Inferência em lote/concorrente dos LLMs locais.
"""

from inference.batch import DEFAULT_BATCH_SIZE, run_batch_inference
from inference.llm_batch import run_llm_batch_inference
from inference.online import OnlinePredictor
from inference.postprocessing import build_prediction_dataframe, standardize_prediction_output
from inference.predictor import Predictor

__all__: list[str] = [
    "DEFAULT_BATCH_SIZE",
    "OnlinePredictor",
    "Predictor",
    "build_prediction_dataframe",
    "run_batch_inference",
    "run_llm_batch_inference",
    "standardize_prediction_output",
]
