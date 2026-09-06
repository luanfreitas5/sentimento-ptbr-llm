# sentimento-ptbr-llm

Estudo comparativo entre **Machine Learning clássico**, **Deep Learning/Transformers** e **LLMs locais** para análise de sentimentos (positivo/negativo/neutro) em tweets em português brasileiro.

## Por que este projeto existe

A maior parte da literatura de análise de sentimentos em pt-BR compara paradigmas isoladamente (só ML clássico, ou só Transformers). Este projeto avalia os **quatro paradigmas lado a lado**, sob o mesmo protocolo experimental — mesmo split, mesmas métricas, mesmos testes estatísticos — para responder de forma defensável: *o ganho de um LLM local ou de um Transformer fine-tuned justifica o custo computacional adicional em relação a um classificador clássico bem ajustado?*

## Paradigmas comparados

| Paradigma | Modelos | Representação | Módulo |
|---|---|---|---|
| ML clássico | Naive Bayes, Regressão Logística, SVM, Random Forest, Gradient Boosting | TF-IDF, FastText, embeddings contextuais + autoencoder | `src/models/*.py` |
| Deep Learning | BiLSTM, CNN para texto | Embeddings estáticos (FastText) | `src/models/lstm.py`, `src/models/cnn.py` |
| Transformers (fine-tuning) | BERTimbau, RoBERTa pt-BR, DistilBERT pt-BR | Embeddings contextuais | `src/models/bertimbau.py`, `roberta.py`, `distilbert.py` |
| LLMs locais | Llama 3.1, Gemma 2 (Ollama/Hugging Face) | Prompting zero-shot / few-shot / chain-of-thought | `src/llm/` |

## Por onde começar

- **Instalar e rodar pela primeira vez** → [Guia de setup](guides/setup.md)
- **Entender os estágios do pipeline (`--stage`)** → [Guia de pipeline](guides/pipeline.md)
- **Desenho experimental e métricas** → [Metodologia](guides/metodologia.md)
- **Seeds, hashes de dados, MLflow, Git SHA** → [Reprodutibilidade](guides/reprodutibilidade.md)
- **Documentação de código (docstrings NumPy em pt-BR)** → [Referência da API](reference.md)

## Documentos de referência

- [`projeto-mestrado-analise-sentimentos-ptbr.md`](https://github.com/luanfreitas5/sentimento-ptbr-llm/blob/main/projeto-mestrado-analise-sentimentos-ptbr.md) — documento mestre do desenho experimental.
- [`PLANO-ELABORACAO.md`](https://github.com/luanfreitas5/sentimento-ptbr-llm/blob/main/PLANO-ELABORACAO.md) — plano de construção do repositório, pasta a pasta.
- [Model Cards](https://github.com/luanfreitas5/sentimento-ptbr-llm/tree/main/reports/model_cards) e [Datasheet do corpus](https://github.com/luanfreitas5/sentimento-ptbr-llm/blob/main/reports/datasheets/datasheet_corpus_tweets.md) — documentação de IA responsável.
