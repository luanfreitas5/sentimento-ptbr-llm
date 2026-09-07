# Processamento paralelo de lotes de tweets brutos (preprocessing, labeling, features)

Data: 2026-09-07

## Contexto e problema

`data/raw/` já contém múltiplos arquivos Parquet reais coletados por usuário
(ex.: `A_Albuquerque99_fulljson.parquet`), cada um com as colunas
`user_id, tweet_id, text, created_at, language, is_reply, is_retweet,
like_count, reply_count, retweet_count, quote_count, source_query,
source_group`.

O pipeline atual, porém, assume um único arquivo consolidado
(`paths.raw_tweets_file`) com um schema mínimo e incompatível
(`id, text, data_source, data_collected`), produzido pela etapa `ingestion`
(scraping ao vivo). As etapas `preprocessing`/`labeling` processam o corpus
inteiro em laços Python sequenciais (uma linha por vez).

Objetivo: atualizar as etapas `preprocessing`, `labeling` e `features` para
consumir diretamente o lote de arquivos já existentes em `data/raw/`,
validá-los contra um contrato de dados que reflita as colunas reais, e
paralelizar o processamento de grandes volumes de tweets, produzindo ao
final um corpus rotulado pronto para a etapa `features` (split e TF-IDF).

## Fora do escopo

- A etapa `ingestion` (scraping ao vivo via `twscrape`) não é alterada: ela
  não valida contra `RawTweetSchema` hoje (apenas grava o que `scrape_func`
  retornar) e continua existindo para coletas futuras, desacoplada deste
  fluxo.
- `pipelines/features.py` não ganha código de paralelismo próprio (ver
  Decisão 5).
- Rotuladores baseados em LLM/modelo de referência (`configs/labeling.yaml`)
  continuam não implementados; a paralelização de rotulagem cobre o
  rotulador heurístico-lexical existente.

## Decisões de design

### 1. Contrato de dados brutos (`schemas/dataset.py`, `constants/columns.py`)

`RawTweetSchema` passa a modelar exatamente as colunas reais de
`data/raw/*.parquet`:

| Campo | Tipo | Observação |
|---|---|---|
| `tweet_id` | `str` | único — chave primária do schema bruto |
| `user_id` | `str` | |
| `text` | `str` | |
| `created_at` | `datetime` | |
| `language` | `str` | código de idioma reportado pela coleta |
| `is_reply` | `bool` | |
| `is_retweet` | `bool` | |
| `like_count`, `reply_count`, `retweet_count`, `quote_count` | `int`, `ge=0` | métricas de engajamento |
| `source_query` | `str` | termo/consulta de coleta |
| `source_group` | `str` | agrupamento de origem (ex.: usuário coletado) |

Os campos antigos (`id`, `data_source`, `data_collected`) são removidos —
nada fora deste schema depende deles (`pipelines/ingestion.py` não valida
contra `RawTweetSchema`). `Config.strict = True` é mantido.

`LabeledCorpusSchema` (`strict = False`) não muda: colunas extras (métricas
de engajamento, `created_at` etc.) seguem intactas até o corpus rotulado,
disponíveis para avaliação por slice no futuro.

`constants/columns.py` ganha as novas constantes de nome de coluna
(`TWEET_ID_COLUMN`, `USER_ID_COLUMN`, `CREATED_AT_COLUMN`,
`LANGUAGE_COLUMN`, `IS_REPLY_COLUMN`, `IS_RETWEET_COLUMN`,
`LIKE_COUNT_COLUMN`, `REPLY_COUNT_COLUMN`, `RETWEET_COUNT_COLUMN`,
`QUOTE_COUNT_COLUMN`, `SOURCE_QUERY_COLUMN`, `SOURCE_GROUP_COLUMN`) e um
novo `RAW_TWEET_BATCH_REQUIRED_COLUMNS`, efetivamente usadas pelo novo
carregador (item 2) — não apenas declaradas. As constantes antigas não
utilizadas (`TEXT_COLUMN = "texto"` etc.) não são tocadas: divergência
pré-existente, fora do escopo desta mudança.

### 2. Carregamento em lote paralelo (`data/loader.py` + `parallel/data_loading.py`)

Novo módulo `src/parallel/data_loading.py`, seguindo o padrão dos demais
módulos de `src/parallel/` (um wrapper fino sobre `execute_parallel_tasks`):

```python
def run_parallel_parquet_loading(
    file_paths: Iterable[Path],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[Path, pl.DataFrame]:
    """Lê múltiplos arquivos Parquet em paralelo (ThreadPoolExecutor)."""
```

Usa `ThreadPoolExecutor`: a leitura/decodificação Parquet do polars é
implementada em Rust e libera o GIL, então threads evitam o custo de
serializar DataFrames inteiros entre processos (mesmo raciocínio já
documentado em `parallel/scraping.py` para I/O).

`data/loader.py` ganha `load_raw_tweet_batch(directory: Path, *,
max_workers=None, show_progress=True) -> pl.DataFrame`, substituindo
`load_raw_tweet_dataset` (arquivo único):

1. `directory.glob("*.parquet")` — levanta `EmptyDatasetError` se vazio.
2. `run_parallel_parquet_loading(...)` sobre os caminhos encontrados; uma
   falha de leitura isolada (arquivo corrompido) é logada e resulta em
   `DataError` acumulado, sem abortar os demais arquivos.
3. `pl.concat(leituras_bem_sucedidas, how="vertical")`.
4. Validação **única** contra o novo `RawTweetSchema`, feita sobre o
   DataFrame já concatenado (não por arquivo) — assim uma duplicata de
   `tweet_id` entre dois arquivos diferentes é detectada pela
   constraint `unique=True`, o que validar por arquivo isoladamente não
   pegaria.
5. Renomeia `tweet_id -> id` **após** a validação, para que
   `labeling/automatic.py` (`id_column="id"`) e o restante do pipeline
   continuem funcionando sem alteração de contrato.

`load_raw_tweet_dataset` (arquivo único, schema antigo) é removida —
YAGNI: nenhum caminho de código real a usa depois desta mudança.

### 3. Preprocessing (`pipelines/preprocessing.py`, `preprocessing/filtering.py`, `preprocessing/pipeline.py`)

`run_preprocessing_stage`:
- Troca `load_raw_tweet_dataset(paths.raw_tweets_file)` por
  `load_raw_tweet_batch(paths.data_raw_dir, max_workers=..., show_progress=...)`.
- Aplica o novo filtro de metadados brutos (abaixo) antes de chamar
  `run_preprocessing_pipeline`.
- Ganha parâmetros `max_workers: int | None = None` e
  `show_progress: bool = True`, repassados a `load_raw_tweet_batch` e a
  `run_preprocessing_pipeline`.

Novo `filter_by_raw_metadata` em `preprocessing/filtering.py`:

```python
def filter_by_raw_metadata(
    dataframe: pl.DataFrame,
    *,
    exclude_retweets: bool = True,
    required_language: str | None = "pt",
) -> pl.DataFrame:
    """Remove retweets e/ou linhas fora do idioma alvo, usando metadados da coleta."""
```

- `exclude_retweets=True` (padrão): remove `is_retweet == True`. Replies
  são mantidos (ainda expressam opinião original do autor).
- `required_language="pt"` (padrão): remove linhas com `language != "pt"`
  — filtro primário, mais barato que a heurística lexical existente
  (`filter_by_portuguese_language`, que continua atuando como filtro
  secundário de qualidade sobre o texto já normalizado).
- Aplicado nesta ordem (retweet, depois idioma) por serem os filtros mais
  baratos, antes do trabalho caro de normalização por linha.

`preprocessing/pipeline.py` — paralelização de `run_preprocessing_pipeline`:

- Novo tipo `_IndexedText = tuple[int, str]` e wrapper de módulo
  `_normalize_indexed_row_text(item: _IndexedText, *, keep_hashtag_word:
  bool) -> _IndexedText`, delegando a `_normalize_row_text` (mantida como
  está, incluindo a conversão de exceção em `PipelineStageError`).
- `run_preprocessing_pipeline` ganha `max_workers: int | None = None` e
  `show_progress: bool = True`.
- O laço serial é substituído por:
  ```python
  indexed_texts = list(enumerate(dataframe[text_column].to_list()))
  resultado = run_parallel_text_cleaning(
      functools.partial(_normalize_indexed_row_text, keep_hashtag_word=keep_hashtag_word),
      indexed_texts,
      max_workers=max_workers,
      show_progress=show_progress,
  )
  if resultado.failures:
      raise resultado.failures[0].error
  normalized_texts = [texto for _, texto in sorted(resultado.successes, key=operator.itemgetter(0))]
  ```
  Reordenar por índice original é obrigatório: `execute_parallel_tasks`
  coleta resultados na ordem de conclusão (`as_completed`), não na ordem
  de submissão — o mesmo problema já resolvido em
  `inference/llm_batch.py` (`_IndexedItem` + `sorted(...,
  key=operator.itemgetter(0))`), replicado aqui pelo mesmo motivo:
  a lista resultante é atribuída de volta como coluna via `pl.Series`,
  então precisa estar alinhada linha a linha com o DataFrame original.
- Falha continua sendo fail-fast (levanta `PipelineStageError` da primeira
  falha), preservando o contrato documentado hoje.
- `run_parallel_text_cleaning` (`parallel/preprocessing.py`) tem sua
  assinatura generalizada de `Callable[[str], str]`/`Iterable[str]` para
  `TypeVar`s genéricos (`ItemType`/`ResultType`), alinhando-a ao estilo já
  usado em `run_parallel_predictions`/`run_parallel_scraping` — sem
  mudança de comportamento, apenas permite reuso com os pares indexados
  acima.

### 4. Labeling (`labeling/automatic.py` + `parallel/labeling.py`)

Novo `src/parallel/labeling.py`, mesmo padrão dos demais wrappers:

```python
def run_parallel_sentiment_labeling(
    label_func: Callable[[ItemType], ResultType],
    items: Iterable[ItemType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função de rotulagem a múltiplos itens em paralelo (ProcessPoolExecutor)."""
```

`ProcessPoolExecutor`: a única implementação concreta de
`SentimentLabeler` hoje (`LexicalHeuristicLabeler`) faz contagem
léxica/regex — CPU-bound, mesmo raciocínio de
`parallel/preprocessing.py`.

`run_cascade_labeling` ganha `max_workers: int | None = None` e
`show_progress: bool = True`. Para cada rotulador da cascata, os textos são
indexados (`list(enumerate(...))`), rotulados em paralelo via
`run_parallel_sentiment_labeling` com um wrapper de módulo
`_label_indexed_item(item: tuple[int, str], *, labeler: SentimentLabeler)
-> tuple[int, str, tuple[str, float]]`, e os resultados reordenados por
`operator.itemgetter(0)` antes de montar as colunas do formato longo —
mesmo padrão indexado do item 3, pelo mesmo motivo (alinhamento linha a
linha, agora por combinação amostra×rotulador).

Comportamento de falha: **fail-fast** (levanta a partir da primeira
`ParallelTaskFailure`), não silenciosamente ignorado — uma falha de
rotulagem não pode resultar em uma linha ausente/incompleta no corpus
rotulado sem sinalização.

`LexicalHeuristicLabeler` é uma classe simples sem estado não serializável
(sem locks, handles de arquivo, threads) — picklable por padrão, compatível
com `ProcessPoolExecutor`.

### 5. Features (`pipelines/features.py`) — sem mudança de código

`run_features_stage` calcula apenas TF-IDF (`sklearn`, já vetorizado em
C) sobre o corpus de treino já particionado; não lê `data/raw` em nenhum
momento. O ganho de paralelismo para "grandes lotes de tweets de usuários"
ocorre integralmente a montante (carregamento, preprocessing, labeling) —
por isso este pipeline não recebe uma implementação própria de paralelismo.
Um parágrafo é adicionado à docstring de `run_features_stage` explicando
esta decisão explicitamente, para não parecer uma lacuna esquecida.

### 6. Configuração de `max_workers`

Reaproveita o `--max-workers` já existente em `src/main.py` (hoje usado por
`ingestion`/`llm_evaluation`/`hypothesaes_analysis`). Nenhuma seção nova em
`configs/config.yaml`. Mudanças em `src/main.py`:

- `_build_preprocessing_stage_kwargs` passa a incluir
  `"max_workers": args.max_workers`.
- `_build_labeling_stage_kwargs` passa a incluir
  `"max_workers": args.max_workers`.
- Docstring do argumento `--max-workers` (linha ~124-128) atualizada para
  listar também `preprocessing`/`labeling`.

## Testes

CLAUDE.md exige cobertura ≥ 80%; as mudanças acima quebram testes
existentes (schema antigo) e introduzem código novo sem cobertura. O plano
de implementação inclui:

- **`tests/test_schemas.py`**: reescrever os casos de `RawTweetSchema` para
  o novo conjunto de colunas (válido, coluna ausente, `tweet_id`
  duplicado, tipo incorreto).
- **`tests/test_data.py`**: substituir `TestLoadRawTweetDataset` (arquivo
  único) por testes de `load_raw_tweet_batch` — diretório com múltiplos
  arquivos válidos concatenados corretamente, diretório vazio
  (`EmptyDatasetError`), um arquivo corrompido isolado sem abortar os
  demais, `tweet_id` duplicado entre dois arquivos rejeitado, renomeação
  `tweet_id -> id` confirmada no resultado.
- **`tests/test_parallel.py`**: testes para `run_parallel_parquet_loading`
  e `run_parallel_sentiment_labeling` (sucesso, falha isolada, respeita
  `max_workers`), seguindo os testes existentes de
  `run_parallel_text_cleaning`/`run_parallel_predictions` como modelo.
- **`tests/test_preprocessing.py`**: teste para `filter_by_raw_metadata`
  (retweet excluído, reply mantido, idioma filtrado, `required_language=None`
  desliga o filtro); teste confirmando que
  `run_preprocessing_pipeline` com `max_workers` produz o mesmo resultado
  (mesma ordem/conteúdo) que a execução serial anterior, e que uma falha
  de normalização ainda levanta `PipelineStageError`.
- **`tests/test_labeling.py`**: teste confirmando que
  `run_cascade_labeling` com `max_workers` produz o mesmo resultado que a
  execução serial, preservando ordem e correspondência amostra↔rótulo.
- Nenhum teste deve depender de dados reais em `data/raw/`; todos usam
  DataFrames/arquivos sintéticos pequenos via `tmp_path`, como já é padrão
  no projeto.

## Addendum (durante o planejamento): pré-requisito de serialização de exceções

Ao detalhar a implementação, foi encontrado um bug pré-existente que bloqueia
diretamente a Decisão 3: toda exceção customizada do projeto herda de
`ProjectError`, mas sobrescreve `__init__` com parâmetros próprios (ex.:
`PipelineStageError(stage_name, detail)`). O pickle padrão do Python
reconstrói uma exceção via `cls(*self.args)`, incompatível com esse padrão —
confirmado empiricamente: levantar `PipelineStageError` dentro de um worker
de `ProcessPoolExecutor` não propaga `PipelineStageError` ao processo pai,
e sim `concurrent.futures.process.BrokenProcessPool`, quebrando o pool
inteiro (violando a garantia de "isolar a falha de um único item" que
`parallel/core.py` documenta e todo o projeto assume).

Corrigido na raiz com um `ProjectError.__reduce__` que reconstrói a
instância via `__new__` (sem chamar o `__init__` da subclasse) — sem
mudança de comportamento para nenhum uso existente, apenas torna toda a
hierarquia de exceções do projeto segura para atravessar um
`ProcessPoolExecutor`. Vira o primeiro item do plano de implementação
(pré-requisito da paralelização de `preprocessing/pipeline.py`).

## Riscos e mitigação

- **Reordenação por conclusão em vez de submissão**: mitigado pelo padrão
  indexado + `sorted(..., key=operator.itemgetter(0))`, já validado em
  produção por `inference/llm_batch.py`.
- **Overhead de `ProcessPoolExecutor` em lotes pequenos** (os 10 arquivos
  atuais somam poucas centenas de linhas): `max_workers=None` deixa o
  executor decidir; não há motivo para forçar paralelismo em datasets
  pequenos — o ganho aparece quando o volume de tweets crescer, que é o
  cenário que motiva esta mudança.
- **`RawTweetSchema` mais rígido pode rejeitar arquivos reais com
  variações de schema** (ex.: coluna ausente em coletas mais antigas): não
  há evidência disso nos 10 arquivos inspecionados (mesmas 13 colunas em
  todos); se surgir, é um problema de dados a tratar caso a caso, não algo
  a acomodar preventivamente no schema.
