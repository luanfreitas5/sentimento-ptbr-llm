# Validade dos rótulos, vazamento e privacidade

Este guia reúne as salvaguardas que não são visíveis nas métricas: o que o rótulo de treino
realmente mede, como o split pode vazar informação e como os dados pessoais dos tweets são
tratados. Os números abaixo foram medidos em 17/11/2026 sobre o corpus de 64.086 tweets.

## 1. O rótulo de treino não é a verdade

Os modelos de `training_*` aprendem o rótulo do classificador do Hugging Face
(`cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`). No teste, as métricas de
`reports/tables/avaliacao/metricas_por_modelo.csv` são medidas contra esse **mesmo** rótulo:
dizem o quanto o modelo reproduz o rotulador, não o quanto acerta o sentimento.

A etapa `evaluate` também mede cada modelo contra referências independentes do rotulador
(`metricas_vs_gold.csv`) e quantifica o ruído do rótulo de treino (`ruido_rotulo_treino.csv`):

| Referência | n | Acurácia do rótulo de treino | Kappa de Cohen | Alfa de Krippendorff |
|---|---|---|---|---|
| `gold_openai` | 1.934 | 0,678 | 0,525 | 0,517 |
| `gold_unb` | 1.926 | 0,629 | 0,459 | 0,436 |
| `consenso_gold` | 1.602 | 0,699 | 0,551 | 0,539 |

Leitura: cerca de 30% dos tweets recebem do rótulo de treino uma classe diferente da dos gold
sets. Um modelo treinado nesse rótulo não deve esperar mais do que essa concordância contra os
gold sets, e o ranking dos modelos muda conforme a referência (ver `metricas_vs_gold.csv`, que
traz IC de bootstrap de 95%).

!!! warning "Os gold sets também são rótulos automáticos"
    `gold_openai` e `gold_unb` são rotulados por LLMs de API. Servem como referência
    independente do rotulador de treino, não como verdade humana. A medida de acerto é a
    validação humana (`make human-sample` e `make human-eval`).

O gold da UnB contém rótulos `indefinido`, que não pertencem às três classes do problema. Eles
são descartados na avaliação (`evaluation.gold_reference.build_gold_references`), porque uma
quarta classe sem suporte nas predições derrubaria o macro-F1 sem relação com o desempenho.

Só os tweets de teste que os gold sets cobrem entram na comparação (392 de 12.817), e nenhum
deles foi usado no treino.

## 2. Comparações múltiplas e tamanho de efeito

A comparação par a par do `evaluate` (`reports/statistics/comparacao_mcnemar.csv`) traz:

- `p_value`, `p_value_holm` e `p_value_bh` (Benjamini-Hochberg); `significant` usa Holm;
- `a_only_correct`, `b_only_correct` e `odds_ratio`: o tamanho do efeito do McNemar;
- `f1_macro_difference` com IC de bootstrap pareado: a diferença na métrica principal.

Com 12.817 tweets de teste, diferenças minúsculas são significativas. Use o IC da diferença de
macro-F1 para decidir se a diferença importa. A validação humana também aplica Holm sobre os
três pares de rotuladores (`p_valor_holm`).

Para validação cruzada, `evaluation.significance.run_corrected_resampled_ttest` implementa o
teste t reamostrado corrigido (Nadeau e Bengio).

## 3. Vazamento entre treino e teste

O CSV bronze identifica o usuário de cada tweet. No split por tweet, **96,4%** dos tweets de
teste têm um autor que também está no treino, e 0,5% dos textos do teste (normalizados)
aparecem no treino.

Para quantificar o efeito, `scripts/experiment_author_leakage.py` treina o mesmo baseline
(TF-IDF + Regressão Logística) nos dois splits:

| Split | Macro-F1 no teste |
|---|---|
| Por tweet (atual) | 0,7243 |
| Por autor (média de 3 sementes) | 0,7233 |

Para este baseline, o vazamento de autor inflou o macro-F1 em cerca de 0,001, ou seja, é
desprezível. O split por autor continua disponível por ser a prática mais segura:

```bash
python scripts/run_tweets_data_pipeline.py --steps pseudonymize-authors
python src/main.py --stage features --group-by-author --drop-cross-split-duplicates
```

A etapa `features` grava `reports/statistics/auditoria_vazamento_split.json` (só proporções)
em toda execução. Para auditar o split já existente sem alterá-lo:

```bash
PYTHONPATH=src python scripts/audit_leakage_and_pii.py
```

!!! note "Mudar o split invalida os resultados anteriores"
    O split por autor gera outros arquivos de treino/validação/teste. Os checkpoints e as
    métricas atuais passam a ser de outra partição, então treine e avalie tudo de novo.

## 4. Privacidade (LGPD)

- **Autores:** o identificador original nunca é gravado. `data/silver/autores_pseudonimizados.parquet`
  guarda só `id` e `author_key`, um HMAC-SHA256 do usuário com a chave `PSEUDONYMIZATION_KEY`
  do `.env`. Sem a chave o pseudônimo não é reversível; com ela, quem conhece um usuário pode
  confirmar se ele está no corpus. Trate a chave como segredo e não a envie ao Git.
- **Textos em relatórios:** toda tabela que sai da camada `data/` passa por
  `utils.privacy.mask_text_columns` (`write_csv(..., mask_text=True)`), que troca @menções,
  URLs, e-mails, CPFs válidos e telefones por marcadores (`@usuario`, `<url>`, `<email>`,
  `<cpf>`, `<telefone>`). O mascaramento é heurístico. Nomes próprios e outros dados
  pessoais no texto livre não são detectados.
- **Auditoria:** `scripts/audit_leakage_and_pii.py` conta, por arquivo, os registros com cada
  categoria de dado pessoal, sem imprimir o conteúdo (`auditoria_dados_pessoais.json`).
- **Colunas de texto cru:** `text` (com @menções e URLs) permanece nas camadas
  `data/silver` e `data/gold`, fora do Git. A coluna `text_normalized` não tem @menções nem
  URLs, mas pode conter números que parecem telefone ou CPF.
- **Base legal e publicação:** documente a base legal e a origem da coleta no datasheet. Se os
  tweets não puderem ser redistribuídos, publique só os IDs de tweet e os hashes do manifesto.
