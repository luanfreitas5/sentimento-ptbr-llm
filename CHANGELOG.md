## v0.4.0 (2026-09-20)

### Feat

- **deps**: Adiciona o ecossistema Jupyter e atualiza as dependências
- **diagnostics**: adicionar pipeline de análise diagnóstica HypotheSAEs
- **diagnostics**: adiciona alvos de diagnóstico e gate de sanidade do HypotheSAEs
- **labeling**: adiciona rastreabilidade completa da origem ao pipeline de rotulagem
- **llm**: Adiciona suporte a múltiplos provedores de LLM
- **llm-relabeling**: Adiciona limitação de taxa para chamadas à API LLM
- **labeling**: adicionar re-rotulagem LLM para amostras de baixa confiança
- **hypotheses**: suporte a métricas de pontuação dinâmicas
- **hypothesaes**: adicionar filtragem de qualidade para interpretações de neurônios
- **preprocessing**: Adição de módulo de lematização e integração com stopwords do NLTK

### Fix

- **prompt**: atualização do formato do modelo de Tweet
- **ci**: evitar conflito de runtime OpenMP em testes no macOS

### Refactor

- **labeling**: substituir o modelo LLM generativo por um classificador de sentimentos otimizado
- **pipeline**: Reestruturação da infraestrutura LLM e do pipeline de rotulagem
- **labeling**: mudança para o pipeline de transformadores do Hugging Face
- **diagnostics**: modernizar a sintaxe e extrair funções auxiliares

## v0.3.0 (2026-09-07)

### Feat

- **hypothesaes**: Implementa a geração de hipóteses via Autoencoder Esparso
- **notebooks**: Adicionar notebooks para análise de classificação de sentimentos
- **app**: adicionar API e camada de implantação de painel
- **pipeline**: Implementar pipeline de dados com execução de estágios e ferramentas de desenvolvimento
- **pipelines**: adicionar orquestração de pipeline de ponta a ponta para análise de sentimentos
- **visualization**: Implementar conjunto de visualização
- **metrics**: adicionar métricas de avaliação de análise de sentimentos
- **experiment**: Adiciona módulo de rastreamento de experimentos e registro de modelos
- **evaluation**: adicionar estrutura de avaliação rigorosa para classificadores de sentimento
- **training**: adicionar módulo de treinamento com callbacks e agendamento
- **llm**: Implementa classificação de sentimentos baseada em LangChain com Ollama/Hugging Face
- **inference**: adicionar camada de inferência unificada com múltiplos modos de predição
- **deps**: Adicionadas dependências para LLM e modelagem estatística
- **models**: adicionar modelos de classificação de sentimentos
- **deps**: adicionar pilha de aprendizado de máquina e ciência de dados
- **hypothesaes**: adicionar módulo de interpretabilidade HypotheSAEs Implementar o pipeline completo do HypotheSAEs para gerar e avaliar hipóteses interpretáveis ​​por meio de Autoencoders Esparsos. Adaptação do método HypotheSAEs (Movva et al.) para o pipeline de análise de sentimentos pt-BR.
- **hypothesaes**: adicionar módulo de interpretabilidade HypotheSAEs
- **preprocessing**: adicionar módulo de pré-processamento para tweets em português
- **labeling**: adicionar módulo de rotulagem de sentimentos em cascata para tweets em português
- **features**: Adicionar módulo de engenharia de recursos para análise de texto em português
- **data**: adicionar um pipeline de dados abrangente com catalogação e validação.
- **parallel**: adiciona uma estrutura genérica de execução paralela
- **core**: Inicializar a infraestrutura do projeto com configuração, utilitários e esquemas
- **config**: adicionar infraestrutura de configuração de pipeline

### Fix

- **core**: aprimoramento da segurança de tipos, robustez e redução de dependências via `pytest -m "not slow"`
- **types**: adicionar comentários de ignorar tipos para importações opcionais e aprimorar dicas de tipo

### Refactor

- melhoria da qualidade do código via `make quality` e padrões pythonicos
- **features,labeling**: : aprimorar a segurança de tipos e a reprodutibilidade do modelo.
- **data**: melhorar a nomenclatura das variáveis ​​para maior clareza
- **parallel**: extrair a coleta de resultados para uma função dedicada
- simplificar o Makefile e padronizar a nomenclatura de variáveis

## v0.2.0 (2026-09-01)

### Feat

- **experiment**: adicionar estrutura de módulo para rastreamento de experimentos
- **visualization**: adicionar estrutura de módulo de visualização
- **project**: inicializa a estrutura do módulo principal
- **inference**: adicionar estrutura de módulo de inferência
- **training**: estrutura básica do módulo de treinamento
- **models**: inicializa o pacote models com arquivos de espaço reservado
- **features**: Criar estrutura de módulo para extração de características
- **labeling**: estrutura básica do módulo de rotulagem
- **preprocessing**: configurar a estrutura do módulo de pré-processamento
- **data**: estrutura do módulo do pipeline de dados
- **parallel**: adiciona estrutura de pacote para módulos de processamento paralelo
- **configs**: adicionar estrutura básica de arquivos de configuração
