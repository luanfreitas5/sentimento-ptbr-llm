# Pipeline

O pipeline é orquestrado por **estágios**, executáveis isoladamente via:

```bash
uv run python src/main.py --stage <nome_do_estagio>
uv run python src/main.py --stage all   # executa todos, na ordem de configs/config.yaml -> stages
```

O acoplamento entre estágios é o sistema de arquivos (Parquet em `data/`, checkpoints em `models/`) — qualquer estágio pode ser reexecutado sem repetir os anteriores, desde que suas entradas já existam em disco. A lista canônica e a ordem dos estágios vivem em `configs/config.yaml -> stages` e no registro `src/pipelines/workflow.py::STAGE_REGISTRY`.

## Estágios

| Estágio | Atalho `make` | Módulo | Entrada | Saída |
|---|---|---|---|---|
| `ingestion` | `pipeline-ingestion` | `src/pipelines/ingestion.py` | função de coleta definida pelo usuário (`--scrape-func`) | `data/interim/tweets_coletados.parquet` |
| `preprocessing` | `pipeline-preprocessing` | `src/pipelines/preprocessing.py` | lote de tweets brutos coletados por usuário, `data/raw/*.parquet` (ver `src/data/loader.py::load_raw_tweet_batch`) | `data/interim/corpus_normalizado.parquet` |
| `labeling` | `pipeline-labeling` (`-huggingface` / `-openai`) | `src/pipelines/labeling.py` | corpus normalizado | `data/processed/tweets_data_huggingface.parquet`, `tweets_data_openai.parquet` (+ `.meta.json`) e `corpus_rotulado.parquet` |
| `comparative_evaluation` | `pipeline-comparative-evaluation` | `src/pipelines/comparative_evaluation.py` | as duas bases rotuladas | tabelas em `reports/tables/comparativo_hf_openai/`, gráficos em `reports/figures/comparativo_hf_openai/`, resumo em `reports/metrics/comparativo_hf_openai.json` |
| `features` | `pipeline-features` | `src/pipelines/features.py` | corpus rotulado | splits treino/validação/teste + features TF-IDF |
| `training_classical` | `pipeline-training-classical` | `src/pipelines/training_classical.py` | splits + features | checkpoints em `models/checkpoints/` |
| `training_deep_learning` | `pipeline-training-deep-learning` | `src/pipelines/training_deep_learning.py` | splits + embeddings | checkpoints (DL/Transformers/autoencoder) |
| `hypothesaes_analysis` | — | `src/pipelines/hypothesaes_analysis.py` | corpus rotulado | diagnóstico de rotulagem em `reports/interpretability/` |

## Rotulagem: duas bases independentes

O estágio `labeling` classifica **todos** os tweets do corpus normalizado com duas fontes, sobre o
texto já sanitizado (`text_normalized`, sem menções/URLs). A fonte OpenAI usa o prompt de
`prompts/` (`configs/labeling.yaml -> prompt_name`); a fonte Hugging Face é um classificador
ajustado e não recebe prompt:

| Base | Fonte | Configuração |
|---|---|---|
| `tweets_data_huggingface` | classificador local via `transformers` (`src/labeling/huggingface.py`) | `configs/labeling.yaml -> huggingface` |
| `tweets_data_openai` | API OpenAI-compatível (`src/labeling/openai_labeler.py`) | `configs/labeling.yaml -> openai`; `OPENAI_BASE_URL`/`OPENAI_KEY` no `.env` |

Colunas (contrato `schemas.labeling.LabeledSourceSchema`): `id`, `text` (texto original),
`text_normalized` (após o pré-processamento), `sentiment_label` (classe atribuída) e
`confidence_score` (confiança). O modelo, o prompt (hash), a temperatura e o hash do corpus de
entrada ficam em `<base>.meta.json`, ao lado da base.

- **Retomada e deduplicação:** cada lote é gravado em `data/interim/labeling_checkpoints/` (JSON
  Lines). Ao reexecutar, tweets já rotulados não são reprocessados; o checkpoint é invalidado se
  modelo, prompt ou temperatura mudarem.
- **Falhas:** erros de API (timeout, HTTP 429) são retentados com backoff exponencial; se sobrar
  tweet sem rótulo válido, a etapa levanta `IncompleteLabelingError` e a base **não** é gravada
  (as duas bases sempre têm exatamente os mesmos tweets). Rode de novo para reprocessar só os pendentes.
- **HTTP 429:** `time.sleep(request_interval_seconds)` antes de cada chamada à API.
- **GPU:** a memória é liberada a cada lote e o modelo é descarregado ao fim da fonte.
- **Modelo Hugging Face:** o padrão é `pysentimento/bertweet-pt-sentiment` (BERTweet-pt, ~135M de
  parâmetros; classes NEG/NEU/POS mapeadas para negativo/neutro/positivo). É um classificador
  ajustado, não um LLM gerativo: não usa prompt, é determinístico e a confiança é a probabilidade
  softmax da classe. Roda em CPU ou GPU pequena. Fixe `huggingface.revision` num SHA.
- **Corpus das etapas seguintes:** `configs/labeling.yaml -> downstream_source` escolhe qual base
  vira `sentiment_label` em `corpus_rotulado.parquet` (consumido por `features` e
  `hypothesaes_analysis`); as duas ficam em `sentiment_label_<fonte>`/`confidence_score_<fonte>`.

```bash
make pipeline-labeling-huggingface   # GPU local
make pipeline-labeling-openai        # API
```

## Avaliação comparativa (`comparative_evaluation`)

Executa diretamente, sem argumentos manuais (`make pipeline-comparative-evaluation`): carrega e
valida as duas bases, une-as pelo `id`, calcula as métricas, grava tabelas e gráficos, analisa as
divergências e, por fim, aplica o HypotheSAEs. Limiares e opções em
`configs/evaluation.yaml -> llm_comparison`; `EXTRA_ARGS=--skip-hypotheses` dispensa o HypotheSAEs.

Sem gold set, mede-se **concordância entre os modelos** (não acerto).

| Análise | Saída (em `reports/tables/comparativo_hf_openai/`) |
|---|---|
| Distribuição das classes | `distribuicao_classes.csv` |
| Concordância/divergência (IC bootstrap), Kappa simples e ponderado, Stuart-Maxwell | `reports/metrics/comparativo_hf_openai.json`, `resumo_comparativo.md` |
| Matriz de concordância | `matriz_concordancia.csv` |
| Confiança (descritivas, Spearman, Wilcoxon pareado) | `confianca_por_modelo.csv` |
| Maiores divergências (polaridade oposta e ambos confiantes) | `maiores_divergencias.csv` |
| Um modelo confiante e o outro não | `conflitos_de_confianca.csv` |
| Tamanho do texto (IC de Wilson) | `concordancia_por_tamanho.csv` |
| Casos ambíguos | `casos_ambiguos.csv`, `resumo_ambiguidade.csv` |
| Diferenças de classificação | `transicoes_divergencia.csv`, `exemplos_transicoes.csv` |

Gráficos (PNG 300 dpi + SVG) em `reports/figures/comparativo_hf_openai/`. As tabelas com texto usam
apenas o texto normalizado.

### HypotheSAEs sobre a divergência

`src/diagnostics/model_disagreement.py` aplica o HypotheSAEs a dois alvos: `disagreement`
(`lab_huggingface != lab_openai`) e `uncertainty` (`1 - min(confiança_HF, confiança_OpenAI)`).
Entrada: texto normalizado + alvo. Processamento: embeddings BERTimbau → SAE (treinado uma vez) →
gate de sanidade → neurônios → hipóteses interpretadas por LLM (`configs/diagnostics.yaml`).
Se o gate reprovar um alvo, isso é registrado como resultado (`gate_reprovado`) — os embeddings
não preveem o alvo acima do acaso —, sem interromper o restante. Saídas: hipóteses
(`reports/interpretability/diagnostics/`) e, por hipótese, os tweets mais ativados
(`hipoteses_<alvo>_evidencias.csv`, com rótulos e confianças dos dois modelos). As hipóteses descrevem
o comportamento dos modelos e devem ser validadas antes de serem tratadas como explicações.

## Estágios que exigem argumentos explícitos

Apenas `ingestion` não tem implementação fixa por design — o projeto não assume uma fonte de scraping:

```bash
# ingestion: --scrape-func aponta para uma função "modulo:funcao" implementada
# por você (ver src/data/downloader.py para o contrato esperado).
make pipeline-ingestion SCRAPE_FUNC=meu_modulo:minha_funcao_de_coleta QUERIES="termo1 termo2"
```

O caminho é resolvido dinamicamente por `src/main.py::_import_callable_from_dotted_path`.
