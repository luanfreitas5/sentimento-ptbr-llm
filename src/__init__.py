"""Código-fonte do projeto ``sentimento-ptbr-llm``.



Packages
--------
config
    Infraestrutura de configuração do projeto.
constants
    Constantes internas do projeto.
data
    Módulo para manipulação de dados.
evaluation
    Avaliação rigorosa de classificadores de sentimento pt-BR.
exceptions
    Exceções customizadas do projeto.
experiments
    Execução de experimentos de treinamento e avaliação de modelos.
features
    Extração de características textuais e vetorização.
hypothesaes
    Geração de hipóteses interpretáveis via Sparse Autoencoders.
inference
    Camada comum de inferência para os quatro paradigmas de modelo do projeto.
io_utils
    Utilitários de entrada/saída para diferentes formatos de arquivo.
labeling
    Anotação de sentimento pt-BR via LLMs (OpenAI ou local).
logging_utils
    Construção de handlers, formatadores e utilitários de log do projeto.
metrics
    Métricas de avaliação de classificadores de sentimento pt-BR.
models
    Implementação de modelos de classificação de sentimento pt-BR.
parallel
    Execução paralela e concorrente de etapas do pipeline.
pipelines
    Pipelines de orquestração ponta a ponta do projeto.
schemas
    Contratos de dados (schemas ``pandera.polars``) do projeto.
training
    Treino de modelos de classificação de sentimento pt-BR.
utils
    Utilitários genéricos do projeto.
visualization
    Visualização de dados, métricas e resultados do projeto.
"""

__all__: list[str] = []
__version__ = "0.2.0"
