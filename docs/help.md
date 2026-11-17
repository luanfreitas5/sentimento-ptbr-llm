usage: main.py [-h]
               --stage {ingestion,preprocessing,labeling,comparative_evaluation,label_consensus,hypotheses,features,training_classical,training_deep_learning,training_transformer,training_llm,evaluate,report,all}
               [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--random-seed RANDOM_SEED] [--max-workers MAX_WORKERS]
               [--portuguese-filter | --no-portuguese-filter]
               [--model-names MODELO1,MODELO2,...] [--track-with-mlflow]
               [--label-source LABEL_SOURCE] [--scrape-func MODULO:FUNCAO]
               [--queries CONSULTA [CONSULTA ...]] [--skip-hypotheses]
               [--force-hypotheses] [--hypotheses-models MODELO [MODELO ...]]
               [--hypotheses-golds GOLD [GOLD ...]]
               [--hypotheses-mode {disagreement,patterns,diagnostics}]
               [--skip-ablation] [--hypothesaes-evaluate] [--dry-run]
               [--diagnostics-step {hypotheses,validation,comparison}]
               [--diagnostics-target {disagreement,uncertainty,pseudo_label,gold_error}]
               [--diagnostics-model-column LAB_MODELO]
               [--diagnostics-label CLASSE]
               [--diagnostics-corpus DIAGNOSTICS_CORPUS]
               [--diagnostics-gold {gold_openai,gold_unb}]

Orquestra as etapas do pipeline de análise de sentimentos pt-BR.

options:
  -h, --help            show this help message and exit

  --stage {ingestion,preprocessing,labeling,comparative_evaluation,label_consensus,hypotheses,features,training_classical,training_deep_learning,training_transformer,training_llm,evaluate,report,all}
                        Nome da etapa a executar (ver `configs/config.yaml ->
                        stages`), ou 'all' para o workflow completo, na ordem
                        configurada.

  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Sobrescreve o nível de log de `configs/logging.yaml`.

  --random-seed RANDOM_SEED
                        Sobrescreve a semente de reprodutibilidade de
                        `configs/config.yaml`.

  --max-workers MAX_WORKERS
                        Número máximo de threads/processos paralelos (etapas
                        `ingestion`/`preprocessing`/`hypotheses`; na etapa
                        `labeling`, sobrescreve `configs/labeling.yaml ->
                        openai.n_workers`, dos gold sets).

  --portuguese-filter, --no-portuguese-filter
                        Aplica (padrão) ou desliga (--no-portuguese-filter) os
                        filtros de idioma português da etapa `preprocessing`
                        (metadado `language` e heurística).

  --model-names MODELO1,MODELO2,...
                        Lista de modelos separados por vírgula (etapas `traini
                        ng_classical`/`training_deep_learning`/`training_trans
                        former`/`training_llm`/`evaluate`); usa o padrão da
                        etapa quando omitido.

  --track-with-mlflow   Habilita o rastreamento MLflow nas etapas de treino.

  --label-source LABEL_SOURCE
                        Fonte(s) de rotulagem da etapa `labeling`:
                        `huggingface` (modelo do Hugging Face, todos os
                        tweets), `huggingface_secondary` (segundo modelo do
                        Hugging Face, todos os tweets), `huggingface_extra`
                        (todos os modelos adicionais de `labeling.yaml ->
                        huggingface_extra`) ou `huggingface_extra:<modelo>`
                        (um só), `gold_openai` (gpt-6-luna) e `gold_unb` (UnB-
                        Llama-3.3-70B-Instruct) — gold sets via API, sobre uma
                        amostra do corpus — ou `all` (todas, em sequência).

  --scrape-func MODULO:FUNCAO
                        Caminho pontilhado para a função de coleta por
                        consulta (etapa `ingestion`), ex.:
                        'meu_pacote.coleta:coletar_por_termo'.

  --queries CONSULTA [CONSULTA ...]
                        Consultas de coleta (etapa `ingestion`), ex.:
                        --queries termo1 termo2.

  --skip-hypotheses     Dispensa a etapa `hypotheses` (útil em `--stage all`
                        sem LLM/GPU).

  --force-hypotheses    Refaz todos os pares da etapa `hypotheses` (modo
                        `disagreement`), ignorando os resumos já concluídos em
                        `reports/metrics/hipoteses_*.json`.

  --hypotheses-models MODELO [MODELO ...]
                        Restringe os modelos do Hugging Face da etapa
                        `hypotheses` (modo `disagreement`): nome completo
                        (ex.: huggingface_extra:FinBERT-PT-BR) ou só o sufixo
                        (FinBERT-PT-BR). Padrão: todos.

  --hypotheses-golds GOLD [GOLD ...]
                        Restringe os gold sets da etapa `hypotheses` (modo
                        `disagreement`), ex.: gold_openai. Padrão: todos.

  --hypotheses-mode {disagreement,patterns,diagnostics}
                        Modo da etapa `hypotheses`: `disagreement`
                        (discordância/incerteza entre o Hugging Face e cada
                        gold set), `patterns` (padrões e inconsistências nos
                        rótulos de baixa confiança) ou `diagnostics` (gate de
                        sanidade, validação e comparação de prompts; ver
                        `--diagnostics-*`).

  --skip-ablation       Dispensa a ablação do pipeline clássico na etapa
                        `evaluate`.

  --hypothesaes-evaluate
                        Habilita a avaliação das hipóteses num holdout via LLM
                        (etapa `hypotheses`, modo `patterns`); sobrescreve
                        `configs/hypothesaes.yaml -> evaluation.enabled`
                        apenas para ligar (nunca desliga).

  --dry-run             Apenas estima nº de chamadas de LLM, tokens e custo,
                        sem rede, embeddings, SAE ou MLflow (etapa
                        `hypotheses`, modo `diagnostics`).

  --diagnostics-step {hypotheses,validation,comparison}
                        Passo do modo `diagnostics` da etapa `hypotheses`:
                        `hypotheses` (gate de sanidade + hipóteses),
                        `validation` (holdout com Bonferroni + amostra para
                        rotulagem) ou `comparison` (prompts v1 vs v2 no gold).

  --diagnostics-target {disagreement,uncertainty,pseudo_label,gold_error}
                        Alvo de diagnóstico (etapa `hypotheses`, modo
                        `diagnostics`, passos `hypotheses`/`validation`).

  --diagnostics-model-column LAB_MODELO
                        Coluna `lab_<modelo>` dos alvos
                        `pseudo_label`/`gold_error`; por padrão, a de
                        `configs/diagnostics.yaml -> targets`.

  --diagnostics-label CLASSE
                        Classe positiva do alvo `pseudo_label` (one-vs-rest);
                        por padrão, a do YAML.

  --diagnostics-corpus DIAGNOSTICS_CORPUS
                        Parquet já no contrato de diagnóstico (obrigatório
                        para `gold_error`: predições sobre o gold); por padrão
                        adapta o corpus rotulado.

  --diagnostics-gold {gold_openai,gold_unb}
                        Gold set do passo `comparison` (etapa `hypotheses`,
                        modo `diagnostics`).
