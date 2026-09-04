"""HypotheSAEs: geração de hipóteses interpretáveis via Sparse Autoencoders.

Porte para este projeto do método HypotheSAEs (Movva et al.), adaptado do
repositório https://github.com/rmovva/HypotheSAEs. Implementa a análise
interpretativa complementar do pipeline de sentimento em pt-BR: treina um
Sparse Autoencoder sobre embeddings dos tweets, seleciona os neurônios mais
preditivos do sentimento (ou de outra variável-alvo), interpreta cada um
como uma hipótese em linguagem natural (via LLM) e avalia estatisticamente
essas hipóteses em um conjunto de dados de holdout — descobrindo, por
exemplo, subgrupos de tweets sistematicamente mal classificados pelos
modelos clássicos/LLM do restante do pipeline.

Fluxo de alto nível (ver ``quickstart.py`` e o notebook original
``quickstart.ipynb`` como referência de uso):

1. ``embedding.py`` calcula embeddings de texto (OpenAI ou modelo local).
2. ``sae.py`` treina o Sparse Autoencoder Top-K sobre esses embeddings.
3. ``select_neurons.py`` seleciona os neurônios mais preditivos do alvo.
4. ``interpret_neurons.py`` interpreta os neurônios selecionados via LLM.
5. ``annotate.py`` / ``evaluation.py`` anotam e avaliam as hipóteses em um
   conjunto de dados real (efeito, significância, calibração).

``quickstart.py`` expõe as quatro funções de mais alto nível
(:func:`train_sae`, :func:`interpret_sae`, :func:`generate_hypotheses`,
:func:`evaluate_hypotheses`), reexportadas neste ``__init__``.

Modules
-------
utils
    Utilitários genéricos: templates de prompt, truncamento de texto, cache
    em JSON.
llm_api
    Cliente OpenAI-compatível (Responses API) para geração de completions.
embedding
    Cálculo e cache de embeddings de texto (OpenAI ou local).
annotate
    Anotação de texto via LLM: verificação de presença/ausência de conceitos.
sae
    Sparse Autoencoder Top-K com perda Matryoshka opcional.
select_neurons
    Seleção dos neurônios do SAE mais preditivos de uma variável-alvo.
interpret_neurons
    Interpretação de neurônios em linguagem natural, via LLM.
evaluation
    Avaliação estatística de hipóteses em um conjunto de dados real.
quickstart
    Funções de alto nível que orquestram o fluxo completo do método.

Dependências opcionais
-----------------------
Este subpacote depende de bibliotecas pesadas ainda não listadas nas
dependências base do projeto: ``torch``, ``openai``, ``tiktoken``,
``scikit-learn``, ``scipy``, ``statsmodels`` e, para embeddings locais,
``sentence-transformers``. Instale-as (ex.: ``uv add torch openai tiktoken
scikit-learn scipy statsmodels sentence-transformers``) antes de usar
qualquer função deste pacote.
"""

from hypothesaes.annotate import annotate_texts_with_concepts
from hypothesaes.embedding import extract_local_embeddings, extract_openai_embeddings
from hypothesaes.evaluation import score_hypotheses
from hypothesaes.interpret_neurons import (
    InterpretConfig,
    LLMConfig,
    NeuronInterpreter,
    SamplingConfig,
    ScoringConfig,
)
from hypothesaes.quickstart import (
    evaluate_hypotheses,
    generate_hypotheses,
    interpret_sae,
    train_sae,
)
from hypothesaes.sae import SparseAutoencoder, load_model
from hypothesaes.select_neurons import select_neurons
from hypothesaes.utils import format_text_for_display

__all__: list[str] = [  # noqa: RUF022 (agrupado por categoria, ver comentários)
    # Funções de fluxo principal
    "train_sae",
    "interpret_sae",
    "generate_hypotheses",
    "evaluate_hypotheses",
    # Classes principais
    "SparseAutoencoder",
    "load_model",
    # Embeddings
    "extract_openai_embeddings",
    "extract_local_embeddings",
    # Interpretação
    "NeuronInterpreter",
    "InterpretConfig",
    "ScoringConfig",
    "LLMConfig",
    "SamplingConfig",
    # Seleção e avaliação
    "select_neurons",
    "score_hypotheses",
    "annotate_texts_with_concepts",
    # Utilitários
    "format_text_for_display",
]
