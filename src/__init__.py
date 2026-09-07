"""Código-fonte do projeto ``sentimento-ptbr-llm``.

Análise de Sentimentos em Textos do Twitter/X em Português Brasileiro: um
estudo comparativo entre algoritmos clássicos de Machine Learning,
arquiteturas de Deep Learning/Transformers e LLMs locais (Ollama/Hugging
Face) para classificação de sentimento (positivo/negativo/neutro) em
tweets em português brasileiro. Inclui pipeline de rotulagem
semiautomática em cascata (com diagnóstico via HypotheSAEs), avaliação
estatisticamente rigorosa (incerteza via bootstrap, testes de McNemar/
Wilcoxon/Friedman, calibração e avaliação por slice) e rastreamento de
experimentos via MLflow. Orquestrado ponta a ponta por ``src/main.py
--stage <nome>``.

Packages
--------
config
    Infraestrutura de configuração do projeto (YAML + ``.env``, caminhos,
    logging e reprodutibilidade).
constants
    Constantes globais reutilizadas por múltiplos módulos.
data
    Ingestão, particionamento e catalogação de dados.
evaluation
    Avaliação rigorosa de classificadores de sentimento pt-BR (incerteza,
    significância estatística, calibração, avaliação por slice e ablation).
exceptions
    Hierarquia de exceções customizadas do projeto.
experiment
    Rastreamento, registro e reprodutibilidade de experimentos (MLflow).
features
    Representações e engenharia de features de texto (TF-IDF, embeddings
    estáticos/contextuais, autoencoder).
hypothesaes
    Geração de hipóteses interpretáveis via Sparse Autoencoders
    (HypotheSAEs), como camada diagnóstica da rotulagem.
inference
    Camada comum de inferência para os quatro paradigmas de modelo do
    projeto (ML clássico, DL, Transformer e LLM).
io_utils
    Utilitários de entrada/saída para diferentes formatos de arquivo.
labeling
    Rotulagem semiautomática em cascata de tweets em português brasileiro.
llm
    Orquestração LangChain de LLMs locais (Ollama/Hugging Face) para
    classificação de sentimento via prompting.
logging_utils
    Construção de handlers, formatadores e utilitários de log do projeto.
metrics
    Métricas de avaliação de classificadores de sentimento pt-BR.
models
    Implementação dos quatro paradigmas de modelo de classificação de
    sentimento pt-BR (ML clássico, deep learning, fine-tuning de
    Transformer e LLM).
parallel
    Execução paralela e concorrente de etapas do pipeline.
pipelines
    Pipelines de orquestração ponta a ponta do projeto.
preprocessing
    Pré-processamento e NLP de tweets em português brasileiro.
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
__version__ = "0.3.0"
