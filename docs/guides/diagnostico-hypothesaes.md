# Diagnóstico HypotheSAEs pós-rotulagem

Camada **opt-in** (`src/diagnostics/`) que explica e ajuda a melhorar a rotulagem por LLMs de
tweets pt-BR. O HypotheSAEs **não classifica**: treina um Sparse Autoencoder (SAE) sobre
embeddings e gera hipóteses em linguagem natural associadas a um **alvo derivado**.
Nada aqui altera o pipeline de rotulagem; `make pipeline-all` continua igual.

## Alvos permitidos

| Alvo | Definição | O que as hipóteses descrevem |
|---|---|---|
| `disagreement` | `lab_a != lab_b` (todos os modelos rotularam) | onde os modelos divergem |
| `uncertainty` | `1 - agreement_score` (contínuo) | onde o consenso é fraco |
| `pseudo_label` | `lab_modelo == classe` (one-vs-rest) | comportamento **do modelo**, não a verdade |
| `gold_error` | `pred != gold` | onde o modelo erra **de fato** |

!!! warning "Limites de interpretação"
    Hipóteses sobre pseudo-rótulo descrevem o comportamento do modelo, nunca a verdade. Somente o
    gold set (TweetSentBR/RePro) ou uma amostra rotulada por humanos mede acerto. Uma hipótese
    sobre discordância diz *onde* os modelos divergem, não *qual* rótulo está certo: as regras do
    prompt v2 exigem revisão humana.

## Contrato de entrada e adaptador

O contrato (`schemas.diagnostics`) é: `id`, `text_normalized`, `agreement_score` (0–1), uma ou
mais colunas `lab_<modelo>` e, opcionalmente, `gold_label`. Enquanto a cascata multi-LLM não
existe, `diagnostics.targets.adapt_labeled_corpus` mapeia o corpus atual
(`sentiment_label_huggingface` → `lab_huggingface`, `sentiment_label_llm_relabel` →
`lab_llm_relabel`, `confidence_score` → `agreement_score`). Só as colunas do contrato passam:
`user_id` e o texto bruto são descartados (LGPD).

!!! note "Viés de seleção no alvo `disagreement`"
    `sentiment_label_llm_relabel` só existe para os candidatos de baixa confiança. O alvo só usa
    tweets rotulados por **todos** os modelos, logo é um subconjunto enviesado do corpus.

## Fluxo

```text
corpus rotulado ──► adapt_labeled_corpus ──► partições disjuntas (treino/validação/teste)
                                              │
                    embeddings (BERTimbau) ◄──┘──► SAE treinado UMA vez (checkpoint versionado)
                                              │
   por alvo:  gate de sanidade (Ridge) ──► generate_hypotheses (treino) ──► estatísticas dos neurônios
                                              │
   validação: top-10 conceitos ──► anotar subamostra do holdout ──► score_hypotheses + Bonferroni
                                              │
   amostra estratificada (~20/conceito) ──► para_rotular.csv ──► rotulagem humana ──► gold_eval
                                              │
   prompt v2 (regras curtas) ──► v1 vs v2 no G_eval (McNemar, Wilcoxon, IC) ──► rerun em G_disc
```

1. **Gate de sanidade** (`diagnostics.sanity`): Ridge nos embeddings → alvo; AUC (binário) ou R²
   (contínuo) no holdout, com IC por bootstrap e teste de permutação. Se o IC inferior não supera
   o acaso ou a permutação não é significativa, o fluxo **aborta** e registra o motivo no MLflow.
2. **Hipóteses** (`diagnostics.hypotheses`): `generate_hypotheses` na partição de descoberta com
   `selection_method="lasso"` e `n_candidate_interpretations=3`, instruções em português.
   A saída traz `hypothesis`, `separation_score`, `regression_pval`, `feature_prevalence` etc.
   Essas três últimas são calculadas sobre as **ativações do SAE** (sem chamadas de LLM); medem o
   neurônio, não a fidelidade da frase.
3. **Validação** (`diagnostics.validation`): anota uma subamostra do holdout e aplica
   `score_hypotheses`; a hipótese sobrevive se `regression_pval < 0,1 / nº de hipóteses`.
4. **Rotulagem humana** (`diagnostics.sampling`, `diagnostics.gold_eval`): o CSV sai **sem
   identificadores**; o mapeamento `sample_id → id` fica em arquivo separado, fora do git.
   Depois de rotulado, MCC e macro-F1 por conceito e por modelo saem com IC bootstrap.
5. **Prompt v2 e comparação** (`diagnostics.prompt_synthesis`, `diagnostics.comparison`).

## Uso

```bash
make install-hypothesaes                    # torch + sentence-transformers + openai
make diag-dry-run TARGET=disagreement       # estima chamadas/custo, sem rede
make diag-hypotheses TARGET=disagreement    # gate + hipóteses (+ MLflow)
make diag-validate TARGET=disagreement      # Bonferroni + reports/tables/para_rotular.csv
make diag-compare GOLD=tweetsentbr          # v1 vs v2 no gold
```

Equivalente sem `make`: `PYTHONPATH=src python -m diagnostics.hypotheses --target disagreement --dry-run`.
`src/` não está no `sys.path` do venv (o projeto usa `package = false`), por isso o `PYTHONPATH=src`.

Também há o estágio `diagnostics` no orquestrador (`src/main.py`), que dispensa o `PYTHONPATH`:

```bash
uv run python src/main.py --stage diagnostics --diagnostics-target disagreement --dry-run
uv run python src/main.py --stage diagnostics --diagnostics-step validation --diagnostics-target disagreement
uv run python src/main.py --stage diagnostics --diagnostics-step comparison --diagnostics-gold tweetsentbr
make pipeline-diagnostics STEP=validation TARGET=disagreement DRY_RUN=--dry-run
```

O estágio é **opt-in**: está registrado em `STAGE_REGISTRY`, mas fora de
`configs/config.yaml -> stages`, então `--stage all` não o executa. No estágio, o MLflow fica
sempre ligado (só o `--dry-run` o desliga); as CLIs `python -m diagnostics.*` têm `--no-mlflow`.

Para `pseudo_label`, informe `--model-column lab_huggingface --label negativo`. Para `gold_error`,
passe `--corpus` com as predições sobre o gold no contrato (geradas por `diag-compare` em
`data/interim/diagnostics/gold_disc_v1.parquet`).

## Configuração e segredos

Tudo em `configs/diagnostics.yaml`, validado por Pydantic (`extra="forbid"`): uma chave
desconhecida falha na inicialização. Segredos **nunca** ficam no YAML:

- `OPENAI_KEY` (e opcionalmente `OPENAI_BASE_URL`) quando `llm.provider: openai`;
- `DIAGNOSTICS_SAMPLE_SALT`: sal do pseudônimo `sample_id`.

## Reprodutibilidade

Seeds fixas, partições determinísticas, SAE com checkpoint por
`(modelo de embedding, M, K, n_treino, seed)`, cache de LLM por hash do conteúdo **completo** da
requisição, e cada execução no MLflow (experimento `sentimento-ptbr-llm/diagnostics`) com alvo,
M, K, método de seleção, modelos, prompt, hash do corpus e Git SHA. O cache do estágio antigo
`hypothesaes_analysis` não distinguia o modelo de embedding; o novo inclui o modelo no nome.

## Custo (ordem de grandeza — valide com `--dry-run`)

O prompt de anotação do HypotheSAEs tem ~650 tokens só de exemplos few-shot, então a fase de
anotação domina o custo. Premissas: ~3,6 mil tweets de descoberta, 20 neurônios × 3 candidatos,
`n_scoring_examples=100`.

| Etapa | Chamadas | Tokens de entrada |
|---|---|---|
| Hipóteses (por alvo) | ≈ 6 mil (60 interpretação + 6 mil anotação) | ≈ 4,6 M |
| Hipóteses (4 alvos) | ≈ 24 mil | ≈ 18 M |
| Validação (10 conceitos × 500 tweets) | 5 mil | ≈ 3,7 M |
| v1 vs v2 (gold ≤ 2 mil × 2 versões × 2 modelos) | ≈ 8 mil | depende do tamanho do prompt v1 |

Total: **≈ 40 mil chamadas e ≈ 30 M de tokens de entrada**. Com Ollama o custo em dinheiro é zero
e o tempo depende do hardware. Com OpenAI, preencha `pricing` no YAML com os preços vigentes
antes de confiar na estimativa em dólares.

## Limitações conhecidas

- **Modelo anotador**: `llama3.2:1b` não segue o formato Yes/No (smoke test: 0 de 6 respostas
  parseáveis; ele repete o prompt). O fluxo aborta se mais de 20 % das respostas não forem
  parseáveis. Use um modelo maior (ex.: `gemma2:9b`) ou a OpenAI.
- **Gold sets**: `data/external/` deve conter `tweetsentbr.parquet`/`repro.parquet`
  (`id`, `text`, `sentiment_label`). Confirme fonte e licença; não há download automático.
- **Nenhum tweet de `G_eval` entra na descoberta**: o gold é dividido em `G_disc` (descoberta do
  alvo `gold_error` e rerun pós-v2) e `G_eval` (reporte v1 vs v2).
- **`prompts/v2.md`**: o carregador de prompts do pipeline de rotulagem só lê `.txt`; a comparação
  lê o v2 direto do caminho configurado. Para usar o v2 na etapa `labeling`, será preciso
  suportar `.md` no carregador (não alterado aqui).
- **Wilcoxon** usa folds pareados do `G_eval` (com `temperature=0` as seeds são idênticas);
  ao menos 6 folds são necessários para alcançar p < 0,05.
