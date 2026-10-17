## v0.5.0 (2026-10-17)

### Feat

- **diagnostics**: substituir R² pela correlação de Spearman para alvos contínuos
- **diagnostics**: migrar Ridge para RidgeCV com busca em grade alpha
- **evaluation**: adiciona métricas de concordância de referência e análise comparativa
- **models**: adicionar mapeamento explícito de nomes de modelos e atualizar a nomenclatura de saída
- **preprocessing**: adicionar opção de filtro para o idioma português
- **pipelines**: adicionar modelo secundário do Hugging Face à avaliação comparativa
- **consensus**: Adicionada a etapa de análise de consenso de rótulos
- **llm**: lidar com respostas LLM truncadas com lógica de repetição adaptativa
- **diagnostics**: Adicionada validação da taxa de falhas de anotação
- **annotate**: preencher anotações ausentes após falhas de LLM
- **llm**: Limitação adaptativa de taxa e ajuste de concorrência de anotações
- **diagnostics**: implementar limitação de taxa para requisições à API OpenAI
- **llm**: suporte a múltiplos provedores LLM com substituição via CLI
- **labeling**: Adicionada a opção de processamento em lote paralelo e configuração do modo de resposta.
- **labeling**: Adicionada limitação de taxa adaptativa e tratamento de sentimentos indefinidos.

### Fix

- **visualization**: Limitar os erros do intervalo de confiança para evitar valores negativos
- **evaluation**: adicionar dicas de tipo para resolver erros do mypy
- **labeling**: fixar a revisão do modelo Hugging Face Cardiff para o SHA de commit específico
- **select_neurons**: substituir o parâmetro de penalidade obsoleto por l1_ratio
- **config**: converte variáveis ​​de ambiente vazias em None para validadores
- **labeling**: correção de erro de digitação no nome do modelo HuggingFace
- **prompt**: atualização do formato do modelo de Tweet

### Refactor

- **diagnostics**: simplificar o pipeline do modelo e adicionar conversão de tipo
- **diagnostics**: extrair instância do RidgeCV para maior clareza
- **models**: substituir SVC por LinearSVC e adicionar calibração de probabilidade
- renomear as configurações do modelo Hugging Face e atualizar o pipeline de rotulagem
- **consensus**: extrair funções de validação e auxiliares
- **pipelines**: simplificar a atribuição de variáveis ​​e as operações com dicionários.
- Reestruturação do pipeline de rotulagem com conjuntos de referência
- **data**: renomear os estágios do pipeline para arquitetura medallion
- reestruturar os estágios do pipeline e aprimorar o gerenciamento de configuração.
- **diagnostics**: simplificar a sintaxe de mesclagem de dicionários e a lógica de comparação
- **llm_api**: extrair a lógica de repetição para uma função dedicada
- **progress**: substituir tqdm por barras de progresso baseadas em rich-based
- **labeling**: extrair a análise e validação de JSON para funções auxiliares
- **labeling**: extrair funções auxiliares para tratamento de erros de limite de taxa e erros de API
- **models**: extrair a construção e serialização de módulos PyTorch
- **config**: esclarecer a configuração do BiLSTM e remover o campo de arquitetura redundante

### Perf

- **annotation**: aumentar os limites de concorrência e otimizar a lógica de repetição

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
