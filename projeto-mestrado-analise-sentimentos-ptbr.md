# Análise de Sentimentos em Textos do Twitter/X em Português Brasileiro: Um Estudo Comparativo entre Algoritmos Clássicos de Machine Learning, Deep Learning e LLMs Locais

**Repositório GitHub**: `sentimento-ptbr-llm`

## 1. Introdução e Justificativa

Redes sociais como o Twitter/X constituem uma das principais fontes de dados textuais em português brasileiro para estudos de opinião pública, monitoramento de marcas, análise de crises e pesquisa sociolinguística computacional. Apesar do volume expressivo de dados disponíveis, o português brasileiro permanece uma língua de recursos limitados (*low-resource*, relativamente à cobertura de benchmarks e modelos em comparação ao inglês), especialmente no que diz respeito a corpora rotulados de sentimento provenientes de mídias sociais — domínio caracterizado por ruído ortográfico, gírias, ironia, emojis e construções sintáticas informais que desafiam tanto abordagens clássicas de NLP quanto modelos de linguagem de propósito geral.

Nos últimos anos, múltiplos paradigmas passaram a coexistir na literatura de análise de sentimentos: (i) pipelines supervisionados tradicionais, baseados em *feature engineering* e modelos de ML clássico; (ii) redes neurais profundas e *fine-tuning* de encoders pré-treinados (ex.: BERTimbau); e (iii) o uso de LLMs via engenharia de prompts (*zero-shot*/*few-shot*), executados localmente por meio de ferramentas como Ollama e modelos abertos disponibilizados no Hugging Face Hub. A execução local é particularmente relevante para dados de redes sociais, que frequentemente envolvem restrições de privacidade e custo que desaconselham o uso de APIs proprietárias na nuvem.

Este projeto propõe uma investigação comparativa sistemática entre esses paradigmas, aplicada à classificação de sentimento (positivo, negativo, neutro) em tweets em português brasileiro, com ênfase em rigor metodológico de avaliação, escalabilidade computacional e reprodutibilidade.

## 2. Objetivos

### 2.1 Objetivo geral
Desenvolver e avaliar comparativamente um pipeline de análise de sentimentos (positivo/negativo/neutro) para tweets em português brasileiro, contrastando algoritmos clássicos de machine learning, arquiteturas de deep learning e transformers, e LLMs locais orientados por engenharia de prompts.

### 2.2 Objetivos específicos
- Construir um corpus de tweets em pt-BR por meio de coleta direta e/ou consolidação de datasets públicos de referência (TweetSentBR, RePro).
- Projetar um pipeline de rotulagem semiautomática em cascata, com pontuação de confiança baseada em concordância entre rotuladores/modelos, para expandir a base rotulada além dos conjuntos-padrão.
- Implementar e comparar três famílias de abordagens de classificação: (a) ML clássico com *features* lexicais/estatísticas e embeddings; (b) deep learning (redes recorrentes e convolucionais) e *fine-tuning* de encoders transformer pré-treinados para pt-BR; (c) LLMs locais via Ollama e Hugging Face com estratégias de *prompt engineering* (zero-shot, few-shot, chain-of-thought).
- Definir um protocolo de avaliação com métricas quantitativas, testes de significância estatística e visualizações que permitam comparação justa entre paradigmas com naturezas distintas (treinado vs. não-treinado).
- Analisar *trade-offs* entre desempenho preditivo, custo computacional e latência de inferência entre as abordagens, com suporte de processamento paralelo para viabilizar a escala do experimento.

## 3. Hipóteses de Pesquisa

- **H1**: Modelos transformer *fine-tuned* (BERTimbau) superam, em Macro F1, tanto o ML clássico com embeddings estáticos/contextuais quanto as arquiteturas de deep learning (LSTM/BiLSTM, CNN) treinadas do zero sobre o mesmo corpus.
- **H2**: LLMs locais via *few-shot prompting* alcançam desempenho estatisticamente não inferior aos modelos *fine-tuned* nas classes positivo/negativo, porém apresentam taxa de erro significativamente maior na classe neutra, dada sua natureza mais ambígua.
- **H3**: Prompts com raciocínio explícito (*chain-of-thought*) melhoram significativamente a acurácia em casos ambíguos (ironia, sentimento misto) em relação ao *zero-shot*, ao custo de maior latência e consumo computacional.
- **H4**: A compressão dos embeddings contextuais via autoencoder mantém desempenho estatisticamente equivalente ao embedding original nos classificadores clássicos, com ganho relevante de eficiência computacional (tempo de treino/inferência).
- **H5**: O pipeline de rotulagem em cascata com pontuação de confiança produz rótulos com concordância (Kappa/Alpha) estatisticamente comparável à anotação de referência dos *gold sets* (TweetSentBR/RePro).
- **H6**: Nenhuma abordagem isolada domina em todas as métricas simultaneamente — espera-se um *trade-off* entre desempenho preditivo (Macro F1/MCC) e custo computacional (tempo de inferência), com LLMs locais favorecidos em cenários de baixo volume/alta necessidade de explicabilidade e modelos *fine-tuned* favorecidos em cenários de alto volume.

## 4. Metodologia

### 4.1 Coleta de dados
- Coleta de tweets em pt-BR via scraping (`twscrape`) e/ou consolidação de datasets públicos existentes.
- Uso de **TweetSentBR** e **RePro** como *gold sets* de referência para validação e calibração do pipeline de rotulagem.
- Documentação do processo de amostragem e dos critérios de inclusão/exclusão (idioma, presença de conteúdo textual mínimo, remoção de retweets puros/spam).

### 4.2 Pré-processamento e NLP
- Normalização específica para o domínio de redes sociais: tratamento de menções, hashtags, URLs, emojis (mapeamento para tokens semânticos ou remoção controlada) e repetições ortográficas informais.
- Tokenização adequada ao português (considerando contrações, gírias regionais e negação, relevante para polaridade).
- Etapas de limpeza documentadas como parte reprodutível do pipeline (não como pré-processamento ad hoc).

### 4.3 Estratégia de rotulagem
- Pipeline de rotulagem em cascata combinando múltiplos rotuladores (modelos e/ou heurísticas), com **pontuação de confiança baseada em concordância** entre eles.
- Uso do [**HypotheSAEs**](https://github.com/rmovva/HypotheSAEs) como camada diagnóstica pós-rotulagem, para identificar padrões sistemáticos de erro ou vieses na rotulagem automática.
- **Identificação de inconsistências**: geração de hipóteses interpretáveis (via HypotheSAEs) sobre as características textuais associadas aos rótulos de baixa confiança/alta discordância, apontando padrões candidatos (ex.: ironia, negação, mistura de sentimentos) a serem priorizados na validação humana.
- Amostragem de subconjuntos para validação humana, permitindo estimar taxa de erro do pipeline de rotulagem antes do uso em treinamento.

### 4.4 Modelagem — Machine Learning Clássico e Representações
- Baselines com representações TF-IDF/bag-of-words + classificadores (Regressão Logística, SVM linear, Naive Bayes multinomial).
- Representações densas de dois tipos, comparadas como entrada para os classificadores tradicionais (Random Forest, Gradient Boosting, SVM):
  - **Embeddings estáticos**: FastText treinado ou pré-treinado em pt-BR.
  - **Embeddings contextuais**: sentence embeddings extraídos de encoders pré-treinados em português (ex.: BERTimbau, Sentence-BERT pt-BR), usados como *features* de entrada.
- **Autoencoder** aplicado sobre os embeddings contextuais, com dois papéis no pipeline:
  - *Redução de dimensionalidade*: compressão do espaço latente antes da alimentação nos classificadores clássicos, avaliando se a representação comprimida preserva (ou melhora) o desempenho frente ao embedding original de alta dimensão.
  - *Diagnóstico não supervisionado*: uso do erro de reconstrução como sinal adicional para identificar tweets atípicos/ambíguos, complementando o HypotheSAEs na camada diagnóstica da seção 4.3.

### 4.5 Modelagem — Deep Learning e Transformers
- Arquiteturas de deep learning clássicas para classificação de texto: redes recorrentes (LSTM/BiLSTM) e redes convolucionais (CNN para texto), treinadas sobre os embeddings da seção 4.4, como ponto intermediário entre ML clássico e transformers.
- *Fine-tuning* de encoders transformer da família BERT pré-treinados/adaptados para português — **BERTimbau, RoBERTa e DistilBERT** —, comparando desempenho preditivo e custo computacional (este último especialmente relevante para o DistilBERT, versão destilada e mais leve).
- Comparação sistemática entre as três camadas de complexidade (ML clássico → deep learning → transformer *fine-tuned* → LLM via prompt), controlando pelo mesmo conjunto de dados e protocolo de avaliação.

### 4.6 LLMs locais (Hugging Face + Ollama) com engenharia de prompts
- Avaliação de LLMs locais disponíveis via **Hugging Face**, executados também via **Ollama** para inferência local.
- Estratégias de *prompt engineering*: classificação zero-shot, few-shot com exemplos balanceados entre classes, e variantes com raciocínio explícito (chain-of-thought) para casos ambíguos (ironia, sentimento misto).
- Padronização de formato de saída via prompts estruturados (ex.: resposta em JSON restrito às três classes), com *parsing* determinístico e tratamento de falhas de formatação.
- Princípio arquitetural: o LLM realiza apenas a tarefa de classificação/explicação sobre o texto de entrada — nenhuma métrica é computada pelo modelo; toda agregação e cálculo estatístico é feita em código Python determinístico.
- Uso do **LangChain** como camada de orquestração sobre os LLMs locais: gerenciamento de *prompt templates* (zero-shot/few-shot/CoT) versionados, encadeamento de etapas (ex.: classificação seguida de justificativa), *output parsers* estruturados (`PydanticOutputParser`/JSON) com *retry* automático em caso de falha de formatação, e integração padronizada com os *backends* Ollama e Hugging Face por meio de uma interface única de LLM.

### 4.7 Programação Paralela
- Paralelização da coleta de dados (scraping assíncrono/concorrente) para reduzir o tempo de construção do corpus.
- Paralelização do pré-processamento e da extração de *features*/embeddings via `multiprocessing`/`joblib`, aproveitando os múltiplos núcleos disponíveis.
- Execução paralela/batched da inferência dos LLMs locais (Ollama) e dos modelos Hugging Face, com controle de concorrência para não saturar a GPU/CPU disponível.
- Paralelização dos experimentos de avaliação (treino e inferência de múltiplos modelos e *seeds*) para viabilizar validação cruzada e testes estatísticos em tempo hábil.

### 4.8 Métricas de Avaliação
- Acurácia, Precisão, Recall e F1-Score (por classe).
- **Macro F1** e **Weighted F1**, dado o desbalanceamento típico entre classes em dados de redes sociais.
- **ROC-AUC** e **PR-AUC** (estratégia one-vs-rest para o cenário multiclasse).
- **MCC (Matthews Correlation Coefficient)**, como métrica robusta a desbalanceamento de classes.
- Métricas operacionais: tempo de inferência e custo computacional (importantes para justificar a viabilidade de LLMs locais frente a modelos *fine-tuned* menores).

### 4.9 Análises Estatísticas
- Teste de significância estatística entre modelos (ex.: teste de McNemar) para comparação par a par.
- **Teste de Wilcoxon** para comparação pareada entre dois modelos ao longo de múltiplas execuções/*folds*.
- **Teste de Friedman** para comparação simultânea entre mais de dois modelos, com *post-hoc* quando aplicável.
- Análise de importância das variáveis (*feature importance*) para os modelos clássicos e de deep learning.
- **Ablation Study**: remoção controlada de grupos de atributos (ex.: sem embeddings contextuais, sem autoencoder, sem etapas de pré-processamento específicas) para isolar a contribuição de cada componente do pipeline.

### 4.10 Visualizações
- Distribuição das classes (positivo/negativo/neutro) no corpus rotulado.
- Word Clouds e frequência de palavras por classe.
- Análise de N-grams mais frequentes/discriminativos por classe.
- Matriz de confusão por abordagem.
- Curvas ROC e Curvas Precision-Recall por modelo.
- **SHAP Values** e **LIME** para interpretabilidade dos modelos clássicos e de deep learning (explicações complementares em nível global e local).
- **UMAP/t-SNE** dos embeddings (estáticos, contextuais e do espaço latente do autoencoder), para inspeção visual da separabilidade entre classes.
- **Gráfico de barras divergentes** das hipóteses interpretáveis geradas pelo HypotheSAEs (seção 4.3), contrastando padrões textuais associados a rótulos de alta vs. baixa confiança.

### 4.11 Rastreamento de Experimentos
- Uso do **MLflow** como camada transversal a todas as fases de modelagem (4.4–4.6), registrando hiperparâmetros, métricas (seção 4.8), artefatos (matrizes de confusão, gráficos SHAP/LIME, curvas ROC/PR) e versionamento de modelos.
- Model Registry do MLflow para versionar os modelos candidatos por abordagem (ML clássico, deep learning, transformer *fine-tuned*) e facilitar a comparação reprodutível entre execuções.
- Execuções (*runs*) organizadas por experimento (ex.: um experimento por família de modelo), permitindo consulta cruzada entre configuração e resultado nas análises estatísticas da seção 4.9.

### 4.12 Documentação de IA Responsável
- **Datasheet for Datasets** (Gebru et al., 2021): documentação estruturada do corpus de tweets, cobrindo motivação, composição, processo de coleta, processo de rotulagem (incluindo os índices de confiança/discordância da seção 4.3) e usos recomendados/desencorajados.
- **Model Card** (Mitchell et al., 2019): documentação padronizada para cada modelo avaliado (ML clássico, deep learning, transformer *fine-tuned*, LLM local), cobrindo detalhes de treinamento/uso pretendido, métricas de desempenho desagregadas por classe (seção 4.8) e limitações conhecidas.

## 5. Cronograma

Cronograma de referência para 12 meses de execução, organizado em fases sequenciais com sobreposição parcial; adaptável ao tempo remanescente do programa de pós-graduação.

| Fase | Atividades | Período |
|---|---|---|
| 1. Revisão bibliográfica e formalização | Revisão da literatura, refinamento das hipóteses, definição do protocolo de avaliação | Mês 1–2 |
| 2. Coleta e pré-processamento de dados | Scraping, consolidação de datasets, normalização e NLP | Mês 2–3 |
| 3. Pipeline de rotulagem | Rotulagem em cascata, validação contra *gold sets*, diagnóstico via HypotheSAEs/autoencoder | Mês 3–4 |
| 4. ML clássico e representações | TF-IDF, embeddings estáticos/contextuais, autoencoder, baselines | Mês 4–5 |
| 5. Deep Learning e Transformers | LSTM/BiLSTM, CNN, *fine-tuning* de BERTimbau/RoBERTa/DistilBERT | Mês 5–7 |
| 6. LLMs locais e engenharia de prompts | Avaliação de LLMs locais via Ollama/Hugging Face (zero-shot, few-shot, CoT) | Mês 7–8 |
| 7. Orquestração com LangChain | Implementação da camada de orquestração (prompt templates, chains, output parsers estruturados) sobre os LLMs locais, com integração padronizada Ollama/Hugging Face | Mês 8–9 |
| 8. Avaliação comparativa | Métricas, testes estatísticos (Wilcoxon/Friedman), *ablation study*, visualizações (SHAP/LIME/UMAP) | Mês 9–10 |
| 9. Redação do artigo de dissertação | Consolidação dos resultados, escrita do artigo científico (formato de artigo de dissertação), revisão e formatação para submissão | Mês 10–12 |
| 10. Qualificação/defesa | Ajustes finais e defesa perante a banca | Mês 12 |

## 6. Stack Tecnológico
- **Repositório**: `sentimento-ptbr-llm` (GitHub).
- **Linguagem/dados**: Python, Polars (priorizado sobre pandas) para manipulação dos dados.
- **Gerenciamento de ambiente**: `uv`.
- **Modelagem**: Hugging Face Transformers (BERTimbau, RoBERTa, DistilBERT), Ollama para inferência local de LLMs, PyTorch para autoencoder, LSTM/BiLSTM e CNN.
- **Orquestração de LLMs**: LangChain para *prompt templates*, encadeamento de etapas (*chains*) e *output parsers* estruturados sobre os *backends* Ollama/Hugging Face.
- **Programação paralela**: `multiprocessing`/`joblib` para CPU-bound, execução assíncrona para coleta de dados e inferência em lote.
- **Rastreamento de experimentos**: MLflow (tracking de parâmetros/métricas/artefatos e Model Registry).
- **Rotulagem/diagnóstico**: pipeline próprio de rotulagem em cascata, HypotheSAEs.
- **Interpretabilidade/visualização**: SHAP, LIME, UMAP/t-SNE, matplotlib/seaborn.
- **Convenções de código**: identificadores em inglês, docstrings em português (formato NumPy), documentação centralizada via `CLAUDE.md`.

## 7. Contribuições Esperadas
- Um corpus rotulado (ou pipeline de rotulagem reprodutível) para sentimento em tweets pt-BR, documentado e validado contra *gold sets* de referência.
- Evidência comparativa sistemática entre ML clássico, deep learning/transformers e LLMs locais, em um domínio textual ruidoso e específico do português brasileiro.
- Um protocolo de avaliação estatisticamente rigoroso (testes de Wilcoxon/Friedman, *ablation study*) e visualmente interpretável (SHAP, LIME, UMAP/t-SNE), aplicável a estudos futuros de NLP em português para tarefas de classificação de texto curto.

---
*Nota: esta descrição consolida as decisões de projeto já definidas (pipeline de rotulagem em cascata, modelos pt-BR selecionados, princípio de separação entre cálculo determinístico e uso do LLM) e pode ser adaptada em estrutura conforme o template de projeto/qualificação exigido pelo programa de pós-graduação.*

---
Observações sobre a implementação do projeto:

O modulo `src/main.py` será responsável por orquestrar todas as etapas do projeto, desde a coleta de dados até a análise dos resultados, garantindo a reprodutibilidade e a consistência do processo, através de argumentos bem definidos, por exemplo, via linha de comando `--stage` que especifica a etapa atual do processo.

A pasta `config` conterá arquivos de configuração em formato YAML, permitindo ajustes flexíveis nos parâmetros do projeto, como palavras-chave, hashtags, modelos de LLMs, métricas de avaliação e outros hiperparâmetros relevantes.

A pasta `reports` armazenará os resultados dos experimentos, incluindo métricas de desempenho, gráficos, tabelas, relatórios detalhados e configurações usadas, salvas em formato de arquivo recomendado, como `.json`, `.md`, `,csv`, `.png`, `.svg`, etc., facilitando a análise comparativa entre diferentes abordagens e modelos.

A pasta `logs` será utilizada para registrar logs detalhados de cada etapa do projeto, incluindo mensagens de erro, avisos, informações de progresso e resultados intermediários, permitindo a rastreabilidade e a depuração eficiente do código usando rich logging.

A pasta `tests` conterá scripts de teste unitário e de integração para validar a funcionalidade de cada módulo, garantindo a robustez e a confiabilidade do código, utilizando frameworks de teste como `pytest` ou `unittest`. Testes automatizados serão executados regularmente para detectar regressões e problemas de compatibilidade. Ferramentas de análise de cobertura de código, como `coverage` > 80%, serão utilizadas para garantir que todas as partes críticas do código sejam testadas adequadamente.

Ferramentas de qualidade de código, como `ruff`, `basedPyright`, `refurb` e outros, serão empregadas para manter um padrão consistente de estilo e legibilidade do código, facilitando a colaboração entre pesquisadores e contribuindo para a manutenção a longo prazo do projeto.

O projeto será estruturado de forma modular, com cada etapa implementada em funções ou classes específicas, facilitando a manutenção, a escalabilidade e a integração de novos métodos ou modelos no futuro. Comentários detalhados serão incluídos em cada função e classe, explicando seu propósito e uso. Documentação detalhada será fornecida para cada módulo, incluindo exemplos de uso e instruções para replicação dos experimentos. Usa-se o padrão de projeto de pipeline para garantir que cada etapa seja executada de forma independente, permitindo a execução seletiva de etapas específicas conforme necessário.

O projeto será versionado utilizando Git, com commits frequentes e mensagens descritivas, garantindo rastreabilidade das alterações e facilitando a colaboração entre pesquisadores. Além disso, serão mantidos registros detalhados dos experimentos, incluindo configurações, resultados e métricas de desempenho, para permitir a análise comparativa e a validação dos modelos desenvolvidos. Workflows de integração contínua (CI) serão configurados para automatizar a execução de testes, validação de código e geração de relatórios, garantindo a qualidade e a consistência do projeto ao longo do tempo.

O projeto será conduzido em conformidade com as diretrizes éticas e regulatórias aplicáveis, garantindo a privacidade e a segurança dos dados dos usuários, bem como a integridade científica e a transparência dos resultados obtidos.

O projeto será documentado de forma abrangente, incluindo a descrição detalhada das metodologias utilizadas, os resultados obtidos, as limitações do estudo e as implicações práticas dos achados, visando contribuir para o avanço do conhecimento na área de saúde mental e processamento de linguagem natural.

O projeto usará linguagens de programação e bibliotecas amplamente adotadas na comunidade científica, como Python, PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers, langchain e outras, garantindo a compatibilidade e a facilidade de integração com outras ferramentas e frameworks.
