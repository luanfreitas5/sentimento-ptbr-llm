# Datasheet (estatísticas) — Corpus de Tweets pt-BR

> Gerado automaticamente pela etapa `report`. Complementa o documento escrito à mão [`datasheet_corpus_tweets.md`](./datasheet_corpus_tweets.md) com as contagens reais das partições. Contém apenas estatísticas agregadas: nenhum texto de tweet, @menção, URL ou identificador (LGPD).

## Partições

| split | n_tweets |
| --- | --- |
| treino | 44860 |
| validacao | 6409 |
| teste | 12817 |

## Distribuição das classes de sentimento

| split | sentiment_label | count | proportion |
| --- | --- | --- | --- |
| treino | negativo | 14489 | 0.323 |
| treino | neutro | 17991 | 0.401 |
| treino | positivo | 12380 | 0.276 |
| validacao | negativo | 2070 | 0.323 |
| validacao | neutro | 2570 | 0.401 |
| validacao | positivo | 1769 | 0.276 |
| teste | negativo | 4140 | 0.323 |
| teste | neutro | 5140 | 0.401 |
| teste | positivo | 3537 | 0.276 |

## Comprimento dos tweets (palavras, texto normalizado)

| split | mean_words | median_words | p95_words |
| --- | --- | --- | --- |
| treino | 14.77 | 11.0 | 45.0 |
| validacao | 14.93 | 11.0 | 45.0 |
| teste | 14.77 | 11.0 | 44.0 |

## Observações

- Split estratificado pela classe, com semente fixa (`configs/config.yaml -> data_split`).
- Rótulos atribuídos por rotuladores automáticos (consenso/LLMs); ver o datasheet manual para a base legal, a coleta e as limitações.
