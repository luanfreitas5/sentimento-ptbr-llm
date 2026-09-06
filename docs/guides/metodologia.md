# Metodologia

Resumo do desenho experimental. A descrição completa está no documento mestre, [`projeto-mestrado-analise-sentimentos-ptbr.md`](https://github.com/luanfreitas5/sentimento-ptbr-llm/blob/main/projeto-mestrado-analise-sentimentos-ptbr.md).

## Problema e variável-alvo

Classificação de sentimento em três classes — `negativo`, `neutro`, `positivo` (`configs/config.yaml -> labels`) — a partir do texto de tweets em português brasileiro.

## Fontes de dados

- **Coleta própria** de tweets via `twscrape` (`src/parallel/scraping.py`, `src/data/downloader.py`), termos de busca informados pelo usuário.
- **Gold sets públicos** para validação/benchmark: [TweetSentBR](https://huggingface.co/datasets) e RePro, carregados em `data/external/`.

## Rotulagem semiautomática em cascata

Implementada em `src/labeling/`: rotuladores automáticos/heurísticos combinados por consenso (`consensus.py`), com pontuação de confiança/discordância por amostra (`confidence.py`). Amostras de baixa confiança são encaminhadas para validação humana (`manual.py`), e o resultado é comparado contra os gold sets via Kappa/Alpha (`validation.py`).

O diagnóstico da qualidade da rotulagem usa **HypotheSAEs** (Autoencoder Esparso — `src/hypothesaes/`, port do [repositório original](https://github.com/rmovva/HypotheSAEs)) para descobrir padrões e hipóteses de inconsistência nos tweets de baixa confiança.

## Representações

| Representação | Uso | Módulo |
|---|---|---|
| TF-IDF / bag-of-words | ML clássico (baseline) | `src/features/`, `configs/model_params.yaml -> classical.tfidf` |
| FastText (estático, pré-treinado pt-BR) | ML clássico, DL (LSTM/CNN) | `configs/model_params.yaml -> embeddings.static` |
| BERTimbau (contextual) | ML clássico sobre embeddings, autoencoder | `configs/model_params.yaml -> embeddings.contextual` |
| Autoencoder (redução de dimensionalidade) | Reduz embeddings contextuais de 768 → 128 dimensões | `configs/model_params.yaml -> autoencoder`, `src/models/autoencoder.py` |

## Paradigmas comparados

1. **ML clássico** — Naive Bayes, Regressão Logística, SVM, Random Forest, Gradient Boosting (`src/models/*.py`).
2. **Deep Learning** — BiLSTM e CNN para texto sobre embeddings estáticos (`src/models/lstm.py`, `cnn.py`).
3. **Transformers (fine-tuning)** — BERTimbau, RoBERTa pt-BR, DistilBERT pt-BR (`src/models/bertimbau.py`, `roberta.py`, `distilbert.py`).
4. **LLMs locais** — Llama 3.1 e Gemma 2 via Ollama/Hugging Face, orquestrados com LangChain, sob três estratégias de prompt: zero-shot, few-shot e chain-of-thought (`src/llm/`, `configs/llm.yaml`).

O LLM realiza apenas classificação/justificativa em linguagem natural — toda agregação e cálculo estatístico é feito em código Python determinístico (`src/evaluation/`), nunca pelo próprio modelo.

## Avaliação

- **Métrica principal**: F1-macro — robusto ao desbalanceamento típico entre as três classes (`configs/evaluation.yaml -> metrics.primary`).
- **Métrica secundária robusta**: MCC (Matthews Correlation Coefficient), além de F1-weighted, acurácia e precisão/recall macro.
- **Incerteza**: intervalos de confiança via bootstrap (1000 reamostragens, IC 95% — `src/evaluation/evaluator.py`), nunca uma métrica pontual isolada.
- **Significância estatística**: McNemar (par a par), Wilcoxon (pareado entre folds), Friedman + post-hoc de Nemenyi (múltiplos modelos simultaneamente) — `src/evaluation/significance.py`.
- **Calibração**: curva de confiabilidade e Brier score, quando probabilidades são usadas para decisão (`src/evaluation/calibration.py`).
- **Avaliação por slice**: métricas por fonte de dados, classe e comprimento do texto — não apenas agregadas (`src/evaluation/slice_evaluation.py`).
- **Ablation study**: remoção controlada de componentes (embeddings contextuais, autoencoder, pré-processamento de emojis, chain-of-thought — `configs/evaluation.yaml -> ablation`).
- **Thresholds de regressão de métrica**: build falha se F1-macro < 0.65 ou MCC < 0.45 (`configs/evaluation.yaml -> regression_thresholds`).

## Interpretabilidade

SHAP e LIME (global e local) para os classificadores clássicos e Transformers; UMAP/t-SNE para visualizar embeddings e o espaço latente do autoencoder; gráfico de barras divergentes para as hipóteses descobertas pelo HypotheSAEs (`src/visualization/`).

## Escopo e limitações declaradas

- Nenhum atributo sensível (idade, gênero, região) é definido no domínio — auditoria de fairness não se aplica a este projeto.
- Sem deploy contínuo em produção neste escopo — monitoramento de drift não é aplicável (`configs/deploy.yaml` mantém API/dashboard desabilitados por padrão).
