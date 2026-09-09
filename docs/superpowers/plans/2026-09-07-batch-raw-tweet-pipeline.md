# Processamento paralelo de lotes de tweets brutos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the `preprocessing`, `labeling` and `features` stages to consume the real batch of per-user Parquet files already in `data/raw/` (validated against their real column contract) and to process large volumes of tweets in parallel, ending in a labeled corpus ready for the `features` stage.

**Architecture:** A new `RawTweetSchema` mirrors the real raw columns; a new parallel loader (`ThreadPoolExecutor`, I/O-bound) reads and concatenates all `data/raw/*.parquet` files in one pass; text normalization and cascade labeling switch from serial Python loops to `ProcessPoolExecutor`-based parallel execution (CPU-bound), using an indexed-item + sort-by-index pattern (already established in `inference/llm_batch.py`) to keep results aligned to their original row. `features` gets no parallel code (already vectorized via sklearn) — see Task 10.

**Tech Stack:** Python 3.13, polars, pandera.polars, concurrent.futures (`ThreadPoolExecutor`/`ProcessPoolExecutor`), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-07-batch-raw-tweet-pipeline-design.md`

## Global Constraints

- Code identifiers in English; docstrings, comments and log messages in pt-BR (NumPy docstring format).
- Type hints required on every function signature.
- No bare `except:`; only catch specific exceptions except where the codebase's established "isolate one item's failure" pattern explicitly uses a broad `except Exception` (already present in `parallel/core.py`, not to be changed).
- Every new/changed function needs a test; overall coverage must stay ≥ 80% (`pyproject.toml` `[tool.coverage.report] fail-under = 80`).
- `pytest` config already sets `pythonpath = ["src"]` — import modules directly (e.g., `from data.loader import ...`), never `from src....`.
- Follow existing project conventions exactly where one already exists for the situation at hand (see Task 1 and Task 6/9 for the indexed-parallel-item convention, copied verbatim from `src/inference/llm_batch.py`).

---

## Task 1: Make `ProjectError` subclasses safe to cross a `ProcessPoolExecutor` boundary

This is a **prerequisite** discovered while validating this plan, not mentioned in the spec: every custom exception in this project subclasses `ProjectError` but overrides `__init__` with its own parameters (e.g. `PipelineStageError(stage_name, detail)`). Python's default exception pickling reconstructs an exception via `cls(*self.args)`, where `self.args` is the single already-formatted message string set by `Exception.__init__`. For `PipelineStageError` this means unpickling calls `PipelineStageError(full_message)` — missing the required `detail` argument — which raises `TypeError` during unpickling.

Verified empirically: raising `PipelineStageError` inside a `ProcessPoolExecutor` worker and calling `future.result()` in the parent does **not** raise `PipelineStageError` — it raises `concurrent.futures.process.BrokenProcessPool` and kills the whole pool, silently breaking the project's "isolate one item's failure" guarantee for every other in-flight task. Task 6 (parallel text normalization) needs to raise `PipelineStageError` from inside a worker on failure, so this must be fixed first.

**Files:**
- Modify: `src/exceptions/base.py`
- Test: `tests/test_exceptions.py`

**Interfaces:**
- Produces: `ProjectError.__reduce__` — no signature change to any existing exception class; purely fixes pickling. All later tasks can raise/catch `PipelineStageError` (or any `ProjectError` subclass) across a `ProcessPoolExecutor` boundary safely.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_exceptions.py`, inside `class TestProjectError:` (after `test_is_exception_subclass`):

```python
    def test_survives_pickling_roundtrip_for_subclass_with_custom_init(self) -> None:
        """Uma subclasse com __init__ próprio (ex.: PipelineStageError) deve sobreviver a pickle/unpickle.

        Necessário para propagar corretamente a exceção original quando ela é
        levantada dentro de um worker de ``ProcessPoolExecutor`` (ver
        ``src/parallel/core.py``): o pickle padrão do Python reconstrói a
        exceção via ``cls(*self.args)``, incompatível com um ``__init__`` que
        não seja ``(message, *, context=None)``.
        """
        import pickle

        from exceptions.pipeline import PipelineStageError

        original = PipelineStageError("normalizacao_texto", "falha ao normalizar")
        restored = pickle.loads(pickle.dumps(original))

        assert isinstance(restored, PipelineStageError)
        assert str(restored) == str(original)
        assert restored.context == original.context
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exceptions.py::TestProjectError::test_survives_pickling_roundtrip_for_subclass_with_custom_init -v`
Expected: FAIL with `TypeError: PipelineStageError.__init__() missing 1 required positional argument: 'detail'`

- [ ] **Step 3: Write minimal implementation**

In `src/exceptions/base.py`, add `__reduce__` to `ProjectError` and a module-level reconstruction helper. Full new file content:

```python
"""Exceção-base do projeto.

Todas as exceções customizadas do projeto devem herdar de :class:`ProjectError`,
permitindo capturar qualquer falha originada internamente com um único
``except ProjectError`` quando apropriado, sem mascarar exceções de terceiros.
"""

from typing import Any


class ProjectError(Exception):
    """Exceção-base para todos os erros customizados do projeto.

    Parameters
    ----------
    message : str
        Mensagem de erro em pt-BR, descrevendo a falha de forma clara.
    context : dict[str, Any] | None, optional
        Informações adicionais de contexto (ex.: caminho de arquivo, nome de
        coluna, etapa do pipeline) úteis para diagnóstico, by default None.

    Examples
    --------
    >>> raise ProjectError("Falha genérica no projeto")
    Traceback (most recent call last):
        ...
    exceptions.base.ProjectError: Falha genérica no projeto
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self._build_full_message())

    def _build_full_message(self) -> str:
        """Monta a mensagem final incluindo o contexto, quando houver.

        Returns
        -------
        str
            Mensagem de erro formatada com o contexto anexado.
        """
        if not self.context:
            return self.message
        context_string = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} (contexto: {context_string})"

    def __reduce__(self) -> tuple[Any, ...]:
        """Permite que subclasses com ``__init__`` próprio sejam serializadas (``pickle``).

        Subclasses de :class:`ProjectError` costumam expor um ``__init__``
        com parâmetros específicos (ex.: ``PipelineStageError(stage_name,
        detail)``), incompatível com a reconstrução padrão do ``pickle``
        (que chamaria ``type(self)(*self.args)`` usando a mensagem já
        formatada de ``Exception.__init__``). Necessário para que uma
        exceção levantada dentro de um worker de ``ProcessPoolExecutor``
        (ver ``src/parallel/core.py``) chegue intacta ao processo pai, em
        vez de quebrar o pool inteiro (``BrokenProcessPool``) durante a
        desserialização.

        Returns
        -------
        tuple[Any, ...]
            Par ``(função_reconstrutora, argumentos)`` usado pelo ``pickle``.
        """
        return (_reconstruct_project_error, (self.__class__, self.message, self.context))


def _reconstruct_project_error(
    error_class: type[ProjectError], message: str, context: dict[str, Any]
) -> ProjectError:
    """Reconstrói uma instância de :class:`ProjectError` (ou subclasse) sem chamar seu ``__init__``.

    Parameters
    ----------
    error_class : type[ProjectError]
        Classe concreta a instanciar (``PipelineStageError``, ``DataError`` etc.).
    message : str
        Mensagem original (sem o contexto formatado), igual a ``self.message``.
    context : dict[str, Any]
        Contexto original, igual a ``self.context``.

    Returns
    -------
    ProjectError
        Instância reconstruída, equivalente à original.
    """
    instance = error_class.__new__(error_class)
    ProjectError.__init__(instance, message, context=context)
    return instance
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exceptions.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add src/exceptions/base.py tests/test_exceptions.py
git commit -m "fix(exceptions): permitir que subclasses de ProjectError sobrevivam a pickle/unpickle

Necessário para levantar exceções tipadas dentro de workers de
ProcessPoolExecutor sem quebrar o pool inteiro (BrokenProcessPool)."
```

---

## Task 2: Rewrite `RawTweetSchema` to match the real `data/raw` columns

Real files (e.g. `data/raw/A_Albuquerque99_fulljson.parquet`) have columns `user_id, tweet_id, text, created_at, language, is_reply, is_retweet, like_count, reply_count, retweet_count, quote_count, source_query, source_group` — confirmed against all 10 files currently in `data/raw/` (9800 rows total, `tweet_id` fully unique across files, `source_query`/`source_group` **entirely null** in every row, `language` spans 32 languages with only ~76.5% `"pt"`).

**Files:**
- Modify: `src/schemas/dataset.py`
- Modify: `src/constants/columns.py`
- Modify: `src/constants/__init__.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `RawTweetSchema` (pandera model) with fields `tweet_id, user_id, text, created_at, language, is_reply, is_retweet, like_count, reply_count, retweet_count, quote_count, source_query, source_group`; `validate_raw_tweet_dataset(dataframe: pl.DataFrame) -> pl.DataFrame` (signature unchanged, now validates the new schema); new constants `TWEET_ID_COLUMN = "tweet_id"`, `IS_RETWEET_COLUMN = "is_retweet"`, `LANGUAGE_COLUMN = "language"` in `constants.columns`, re-exported from `constants`. Task 4 and Task 5 import these.

- [ ] **Step 1: Write the failing tests**

Replace the raw-tweet-related tests in `tests/test_schemas.py`. First add the import and a module-level fixture helper near the top of the file (after the existing imports):

```python
from datetime import datetime

import polars as pl
import pytest

from exceptions.data import DataValidationError
from schemas.dataset import validate_labeled_corpus, validate_raw_tweet_dataset
from schemas.experiment import validate_experiment_run_metric
from schemas.labeling import validate_labeling_result
from schemas.prediction import validate_prediction
from schemas.training import validate_training_example


def _minimal_raw_tweet_frame(tweet_ids: list[str] | None = None) -> pl.DataFrame:
    """Constrói um DataFrame mínimo válido contra o novo ``RawTweetSchema``, para testes."""
    ids = tweet_ids if tweet_ids is not None else ["1"]
    n = len(ids)
    return pl.DataFrame(
        {
            "tweet_id": ids,
            "user_id": ["u1"] * n,
            "text": [f"texto {tweet_id}" for tweet_id in ids],
            "created_at": [datetime(2026, 1, 1)] * n,
            "language": ["pt"] * n,
            "is_reply": [False] * n,
            "is_retweet": [False] * n,
            "like_count": [0] * n,
            "reply_count": [0] * n,
            "retweet_count": [0] * n,
            "quote_count": [0] * n,
            "source_query": [None] * n,
            "source_group": [None] * n,
        }
    )
```

Then replace the three existing raw-tweet tests inside `class TestDatasetSchemas:` (`test_validate_raw_tweet_dataset_accepts_valid_dataframe`, `test_validate_raw_tweet_dataset_rejects_extra_column`, `test_validate_raw_tweet_dataset_rejects_duplicate_id`) with:

```python
def test_validate_raw_tweet_dataset_accepts_valid_dataframe(self) -> None:
    """Um DataFrame com todas as colunas obrigatórias e tweet_id único deve ser aceito."""
    result = validate_raw_tweet_dataset(_minimal_raw_tweet_frame(["1", "2"]))
    assert result.height == 2


def test_validate_raw_tweet_dataset_allows_null_source_query_and_group(self) -> None:
    """source_query/source_group nulos (comum quando a coleta é por usuário, não por termo) são aceitos."""
    result = validate_raw_tweet_dataset(_minimal_raw_tweet_frame())
    assert result["source_query"].null_count() == 1
    assert result["source_group"].null_count() == 1


def test_validate_raw_tweet_dataset_rejects_extra_column(self) -> None:
    """Uma coluna extra não declarada deve ser rejeitada (schema strict)."""
    df = _minimal_raw_tweet_frame().with_columns(pl.lit("valor").alias("extra_column"))
    with pytest.raises(DataValidationError):
        validate_raw_tweet_dataset(df)


def test_validate_raw_tweet_dataset_rejects_duplicate_tweet_id(self) -> None:
    """tweet_id duplicado deve violar a restrição de unicidade."""
    with pytest.raises(DataValidationError):
        validate_raw_tweet_dataset(_minimal_raw_tweet_frame(["1", "1"]))


def test_validate_raw_tweet_dataset_rejects_negative_engagement_count(self) -> None:
    """Uma contagem de engajamento negativa deve violar o contrato (like_count >= 0)."""
    df = _minimal_raw_tweet_frame().with_columns(pl.Series("like_count", [-1]))
    with pytest.raises(DataValidationError):
        validate_raw_tweet_dataset(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py::TestDatasetSchemas -v`
Expected: FAIL — `_minimal_raw_tweet_frame` columns don't match the current `RawTweetSchema` (still expects `id`, `data_source`, `data_collected`), so every test raises `DataValidationError` (including the "accepts" ones) or the extra/duplicate tests raise for the wrong reason.

- [ ] **Step 3: Write minimal implementation**

In `src/schemas/dataset.py`, add the import and replace `RawTweetSchema` and `validate_raw_tweet_dataset`'s docstring example:

```python
from datetime import datetime

import pandera.polars as pa
import polars as pl
from pandera.api.polars.model_config import BaseConfig
from pandera.errors import SchemaError
from pandera.typing.polars import Series

from constants.labels import SENTIMENT_CLASSES
from exceptions.data import DataValidationError


class RawTweetSchema(pa.DataFrameModel):
    """Contrato de dados para tweets brutos coletados por usuário (``data/raw``).

    Reflete as colunas reais dos arquivos Parquet coletados via
    ``twscrape`` (um arquivo por usuário). ``source_query``/``source_group``
    são nulos quando a coleta foi feita por usuário, não por termo de
    busca. Validado sobre o lote já concatenado por
    :func:`data.loader.load_raw_tweet_batch`, que em seguida renomeia
    ``tweet_id`` para ``id`` (contrato usado pelo restante do pipeline).
    """

    tweet_id: Series[str] = pa.Field(unique=True)
    user_id: Series[str]
    text: Series[str]
    created_at: Series[datetime]
    language: Series[str]
    is_reply: Series[bool]
    is_retweet: Series[bool]
    like_count: Series[int] = pa.Field(ge=0)
    reply_count: Series[int] = pa.Field(ge=0)
    retweet_count: Series[int] = pa.Field(ge=0)
    quote_count: Series[int] = pa.Field(ge=0)
    source_query: Series[str] = pa.Field(nullable=True)
    source_group: Series[str] = pa.Field(nullable=True)

    class Config(BaseConfig):
        """Configuração do schema: rejeita colunas não declaradas."""

        strict = True
```

Update `validate_raw_tweet_dataset`'s docstring `Examples` section (keep the function body — `try/except SchemaError -> DataValidationError` — unchanged):

```python
    Examples
    --------
    >>> from datetime import datetime
    >>> df = pl.DataFrame(
    ...     {
    ...         "tweet_id": ["1"],
    ...         "user_id": ["u1"],
    ...         "text": ["ótimo produto"],
    ...         "created_at": [datetime(2026, 1, 1)],
    ...         "language": ["pt"],
    ...         "is_reply": [False],
    ...         "is_retweet": [False],
    ...         "like_count": [0],
    ...         "reply_count": [0],
    ...         "retweet_count": [0],
    ...         "quote_count": [0],
    ...         "source_query": [None],
    ...         "source_group": [None],
    ...     }
    ... )
    >>> validate_raw_tweet_dataset(df).height
    1
    """
```

In `src/constants/columns.py`, add near the end of the file (after `LABELED_CORPUS_REQUIRED_COLUMNS`, don't touch anything above it):

```python
# Colunas do lote bruto real coletado por usuário (data/raw/*.parquet),
# usadas por data.loader.load_raw_tweet_batch e
# preprocessing.filtering.filter_by_raw_metadata.
TWEET_ID_COLUMN = "tweet_id"
IS_RETWEET_COLUMN = "is_retweet"
LANGUAGE_COLUMN = "language"
```

In `src/constants/__init__.py`, add the three names to the `from constants.columns import (...)` block (alphabetically) and to `__all__` (alphabetically):

```python
from constants.columns import (
    COLLECTION_DATE_COLUMN,
    CONFIDENCE_COLUMN,
    ID_COLUMN,
    IS_RETWEET_COLUMN,
    LABELED_CORPUS_REQUIRED_COLUMNS,
    LABELER_COLUMN,
    LABELER_WEIGHT_COLUMN,
    LANGUAGE_COLUMN,
    PREDICTED_LABEL_COLUMN,
    RAW_CORPUS_REQUIRED_COLUMNS,
    SOURCE_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TEXT_NORMALIZED_COLUMN,
    TWEET_ID_COLUMN,
)
```

And in `__all__`, insert `"IS_RETWEET_COLUMN"` right after `"ID_TO_LABEL"`, `"LANGUAGE_COLUMN"` right after `"LABEL_TO_ID"`, and `"TWEET_ID_COLUMN"` right after `"TEXT_NORMALIZED_COLUMN"` (keeping the list alphabetically sorted, matching the existing style).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py tests/test_constants.py -v`
Expected: PASS (all tests, including `test_constants.py`'s existing assertions about `RAW_CORPUS_REQUIRED_COLUMNS`, which is untouched)

- [ ] **Step 5: Commit**

```bash
git add src/schemas/dataset.py src/constants/columns.py src/constants/__init__.py tests/test_schemas.py
git commit -m "feat(schemas): RawTweetSchema reflete as colunas reais de data/raw

Substitui o contrato mínimo (id/text/data_source/data_collected) pelo
schema real coletado por usuário via twscrape, com nulidade correta em
source_query/source_group (confirmada contra os arquivos reais em
data/raw/)."
```

---

## Task 3: `parallel/data_loading.py` — parallel Parquet batch reading

**Files:**
- Create: `src/parallel/data_loading.py`
- Test: `tests/test_parallel.py`

**Interfaces:**
- Consumes: `io_utils.parquet.read_parquet(file_path: Path, **kwargs) -> pl.DataFrame` (existing), `parallel.core.execute_parallel_tasks` / `ParallelExecutionResult` (existing).
- Produces: `run_parallel_parquet_loading(file_paths: Iterable[Path], *, max_workers: int | None = None, show_progress: bool = True) -> ParallelExecutionResult[Path, pl.DataFrame]`. Task 4 consumes this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`, a new import and test class at the end of the file:

```python
from pathlib import Path

from io_utils.parquet import write_parquet
from parallel.data_loading import run_parallel_parquet_loading
```

(Add `from pathlib import Path` and `from io_utils.parquet import write_parquet` to the existing import block at the top of the file; add `from parallel.data_loading import run_parallel_parquet_loading` alongside the other `parallel.*` imports.)

```python
class TestRunParallelParquetLoading:
    """Testes da leitura paralela de lote Parquet (``parallel.data_loading``)."""

    def test_reads_all_files_successfully(self, tmp_path: Path) -> None:
        """Todos os arquivos devem ser lidos com sucesso quando não há erro."""
        file_a = tmp_path / "a.parquet"
        file_b = tmp_path / "b.parquet"
        write_parquet(pl.DataFrame({"valor": [1]}), file_a)
        write_parquet(pl.DataFrame({"valor": [2]}), file_b)

        result = run_parallel_parquet_loading([file_a, file_b], show_progress=False, max_workers=2)

        assert result.failures == []
        valores = sorted(df["valor"].to_list()[0] for df in result.successes)
        assert valores == [1, 2]

    def test_isolates_failure_for_missing_file(self, tmp_path: Path) -> None:
        """A falha de leitura de um arquivo não deve interromper a leitura dos demais."""
        file_ok = tmp_path / "ok.parquet"
        write_parquet(pl.DataFrame({"valor": [1]}), file_ok)
        missing_file = tmp_path / "inexistente.parquet"

        result = run_parallel_parquet_loading(
            [file_ok, missing_file], show_progress=False, max_workers=2
        )

        assert len(result.successes) == 1
        assert len(result.failures) == 1
        assert result.failures[0].item == missing_file
```

`pytest` is already imported at the top of `tests/test_parallel.py`. `polars` is **not** currently imported there — add `import polars as pl` alongside the new `from pathlib import Path` and `from io_utils.parquet import write_parquet` imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parallel.py::TestRunParallelParquetLoading -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parallel.data_loading'`

- [ ] **Step 3: Write minimal implementation**

Create `src/parallel/data_loading.py`:

```python
"""Paralelização da leitura de lotes de arquivos Parquet.

Usa múltiplas threads (``ThreadPoolExecutor``): a leitura/decodificação
Parquet do ``polars`` é implementada em Rust e libera o GIL, então threads
evitam o custo de serializar DataFrames inteiros entre processos ao
carregar o lote de tweets brutos coletados por usuário (um arquivo por
usuário — ver ``data/raw/`` e ``src/data/loader.py``).
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from io_utils.parquet import read_parquet
from parallel.core import ParallelExecutionResult, execute_parallel_tasks


def run_parallel_parquet_loading(
    file_paths: Iterable[Path],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[Path, pl.DataFrame]:
    """Lê múltiplos arquivos Parquet em paralelo, isolando a falha de cada arquivo.

    Parameters
    ----------
    file_paths : Iterable[Path]
        Caminhos dos arquivos Parquet a serem lidos.
    max_workers : int | None, optional
        Número máximo de threads usadas, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[Path, pl.DataFrame]
        DataFrames lidos com sucesso e falhas isoladas por arquivo, cada
        uma preservando o caminho que causou o erro.

    Examples
    --------
    >>> run_parallel_parquet_loading([Path("data/raw/usuario1.parquet")])  # doctest: +SKIP
    """
    return execute_parallel_tasks(
        read_parquet,
        file_paths,
        executor_class=ThreadPoolExecutor,
        max_workers=max_workers,
        task_description="Leitura paralela de lote Parquet",
        show_progress=show_progress,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parallel.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/parallel/data_loading.py tests/test_parallel.py
git commit -m "feat(parallel): adicionar leitura paralela de lote Parquet

Nova run_parallel_parquet_loading (ThreadPoolExecutor), usada para
carregar em paralelo os múltiplos arquivos por usuário de data/raw/."
```

---

## Task 4: `data/loader.py::load_raw_tweet_batch` — replace single-file loader

**Files:**
- Modify: `src/data/loader.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `RawTweetSchema`/`validate_raw_tweet_dataset` (Task 2), `run_parallel_parquet_loading` (Task 3), `constants.columns.TWEET_ID_COLUMN`/`ID_COLUMN` (Task 2 / existing).
- Produces: `load_raw_tweet_batch(directory: Path, *, max_workers: int | None = None, show_progress: bool = True) -> pl.DataFrame`, replacing `load_raw_tweet_dataset(file_path: Path) -> pl.DataFrame` (removed). Returns a DataFrame with column `id` (renamed from `tweet_id`) plus all other `RawTweetSchema` columns. Task 7 consumes this.

- [ ] **Step 1: Write the failing tests**

In `tests/test_data.py`, add `from datetime import datetime` to the imports, replace `load_raw_tweet_dataset` with `load_raw_tweet_batch` in the `from data.loader import (...)` block, and add a module-level helper near the top (after the existing `_scrape_*` helpers):

```python
def _build_raw_tweet_batch(tweet_ids: list[str], *, language: str = "pt") -> pl.DataFrame:
    """Constrói um DataFrame de tweets brutos válido, para testes de carregamento em lote."""
    n = len(tweet_ids)
    return pl.DataFrame(
        {
            "tweet_id": tweet_ids,
            "user_id": ["u1"] * n,
            "text": [f"texto {tweet_id}" for tweet_id in tweet_ids],
            "created_at": [datetime(2026, 1, 1)] * n,
            "language": [language] * n,
            "is_reply": [False] * n,
            "is_retweet": [False] * n,
            "like_count": [0] * n,
            "reply_count": [0] * n,
            "retweet_count": [0] * n,
            "quote_count": [0] * n,
            "source_query": [None] * n,
            "source_group": [None] * n,
        }
    )
```

Replace the whole `class TestLoadRawTweetDataset:` block with:

```python
class TestLoadRawTweetBatch:
    """Testes de carregamento em lote e validação de tweets brutos (múltiplos arquivos)."""

    def test_concatenates_and_validates_all_files(self, tmp_path: Path) -> None:
        """Deve concatenar e validar todos os arquivos do diretório, renomeando tweet_id para id."""
        write_parquet(_build_raw_tweet_batch(["1", "2"]), tmp_path / "usuario_a.parquet")
        write_parquet(_build_raw_tweet_batch(["3"]), tmp_path / "usuario_b.parquet")

        result = load_raw_tweet_batch(tmp_path, show_progress=False, max_workers=2)

        assert result.height == 3
        assert "id" in result.columns
        assert "tweet_id" not in result.columns
        assert sorted(result["id"].to_list()) == ["1", "2", "3"]

    def test_raises_for_empty_directory(self, tmp_path: Path) -> None:
        """Um diretório sem nenhum arquivo Parquet deve levantar EmptyDatasetError."""
        with pytest.raises(EmptyDatasetError):
            load_raw_tweet_batch(tmp_path, show_progress=False)

    def test_isolates_corrupted_file_and_validates_remaining_batch(self, tmp_path: Path) -> None:
        """Um arquivo corrompido não deve impedir a validação dos demais arquivos do lote."""
        write_parquet(_build_raw_tweet_batch(["1", "2"]), tmp_path / "usuario_valido.parquet")
        (tmp_path / "usuario_corrompido.parquet").write_bytes(b"nao e um parquet valido")

        result = load_raw_tweet_batch(tmp_path, show_progress=False, max_workers=2)

        assert result.height == 2

    def test_rejects_duplicate_tweet_id_across_files(self, tmp_path: Path) -> None:
        """Um tweet_id duplicado entre dois arquivos diferentes deve violar a unicidade do schema."""
        write_parquet(_build_raw_tweet_batch(["1"]), tmp_path / "usuario_a.parquet")
        write_parquet(_build_raw_tweet_batch(["1"]), tmp_path / "usuario_b.parquet")

        with pytest.raises(DataValidationError):
            load_raw_tweet_batch(tmp_path, show_progress=False, max_workers=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py::TestLoadRawTweetBatch -v`
Expected: FAIL with `ImportError: cannot import name 'load_raw_tweet_batch' from 'data.loader'`

- [ ] **Step 3: Write minimal implementation**

In `src/data/loader.py`, update imports and replace `load_raw_tweet_dataset` with `load_raw_tweet_batch`:

```python
import logging
from pathlib import Path

import polars as pl

from constants.columns import ID_COLUMN, TWEET_ID_COLUMN
from exceptions.data import DataError
from io_utils.csv import read_csv
from io_utils.parquet import read_parquet
from parallel.data_loading import run_parallel_parquet_loading
from schemas.dataset import validate_labeled_corpus, validate_raw_tweet_dataset
from schemas.training import validate_training_example
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_PARQUET_SUFFIXES = frozenset({".parquet"})
_CSV_SUFFIXES = frozenset({".csv"})
```

(`read_dataset_file` stays exactly as-is below this.) Then remove the whole `load_raw_tweet_dataset` function and add in its place:

```python
def load_raw_tweet_batch(
    directory: Path, *, max_workers: int | None = None, show_progress: bool = True
) -> pl.DataFrame:
    """Carrega, concatena e valida o lote de tweets brutos coletados por usuário.

    Lê em paralelo todos os arquivos ``*.parquet`` do diretório informado
    (um por usuário coletado — ver ``data/raw/``), isolando a falha de
    leitura de um arquivo corrompido sem abortar os demais (ver
    :func:`parallel.data_loading.run_parallel_parquet_loading`). A
    validação contra :class:`schemas.dataset.RawTweetSchema` ocorre uma
    única vez, sobre o lote já concatenado — não por arquivo — de forma
    que um ``tweet_id`` duplicado entre dois arquivos diferentes também
    seja detectado.

    Parameters
    ----------
    directory : Path
        Diretório contendo os arquivos Parquet brutos (``paths.data_raw_dir``).
    max_workers : int | None, optional
        Repassado a :func:`parallel.data_loading.run_parallel_parquet_loading`,
        by default None (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default True.

    Returns
    -------
    pl.DataFrame
        Lote de tweets brutos validado, com ``tweet_id`` renomeado para
        ``id`` (contrato usado pelo restante do pipeline).

    Raises
    ------
    EmptyDatasetError
        Se o diretório não contiver nenhum arquivo ``*.parquet``, ou se
        todos os arquivos encontrados falharem na leitura.
    DataValidationError
        Se o lote concatenado violar o contrato de dados.

    Examples
    --------
    >>> load_raw_tweet_batch(Path("data/raw"))  # doctest: +SKIP
    """
    file_paths = sorted(directory.glob("*.parquet"))
    validate_not_empty_collection(file_paths, collection_name=str(directory))

    loading_result = run_parallel_parquet_loading(
        file_paths, max_workers=max_workers, show_progress=show_progress
    )
    for failure in loading_result.failures:
        logger.warning(
            "Falha ao ler arquivo Parquet do lote bruto: %s (%s)", failure.item, failure.error
        )
    validate_not_empty_collection(loading_result.successes, collection_name=str(directory))

    raw_batch = pl.concat(loading_result.successes, how="vertical")
    validated_batch = validate_raw_tweet_dataset(raw_batch).rename({TWEET_ID_COLUMN: ID_COLUMN})

    logger.info(
        "Lote de tweets brutos carregado: %d/%d arquivo(s), %d linha(s) (%s).",
        len(loading_result.successes),
        len(file_paths),
        validated_batch.height,
        directory,
    )
    return validated_batch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/data/loader.py tests/test_data.py
git commit -m "feat(data): substituir load_raw_tweet_dataset por load_raw_tweet_batch

Carrega em paralelo todos os *.parquet de um diretório (um por usuário
coletado), concatena, valida contra o novo RawTweetSchema uma única vez
(pegando duplicatas de tweet_id entre arquivos) e renomeia para id."
```

---

## Task 5: `preprocessing/filtering.py::filter_by_raw_metadata` — retweet/language filter

**Files:**
- Modify: `src/preprocessing/filtering.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `constants.columns.IS_RETWEET_COLUMN`, `constants.columns.LANGUAGE_COLUMN` (Task 2).
- Produces: `filter_by_raw_metadata(dataframe: pl.DataFrame, *, exclude_retweets: bool = True, required_language: str | None = "pt") -> pl.DataFrame`. Task 7 consumes this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_preprocessing.py` (find the existing import block for `preprocessing.filtering` and add `filter_by_raw_metadata` to it), then add a new test class after `class TestFilterByInclusionCriteria:` (before `class TestNormalizeTweetText:`):

```python
class TestFilterByRawMetadata:
    """Testes do filtro de metadados brutos (retweet/idioma) em nível de DataFrame."""

    def test_excludes_retweets_by_default(self) -> None:
        """Por padrão, linhas com is_retweet=True devem ser removidas."""
        df = pl.DataFrame(
            {
                "text": ["a", "b"],
                "is_retweet": [True, False],
                "language": ["pt", "pt"],
            }
        )
        result = filter_by_raw_metadata(df)
        assert result["text"].to_list() == ["b"]

    def test_keeps_replies(self) -> None:
        """Replies (is_reply=True) não devem ser removidas pelo filtro de retweet."""
        df = pl.DataFrame(
            {
                "text": ["a"],
                "is_retweet": [False],
                "language": ["pt"],
                "is_reply": [True],
            }
        )
        result = filter_by_raw_metadata(df)
        assert result.height == 1

    def test_filters_by_required_language(self) -> None:
        """Linhas fora do idioma exigido devem ser removidas."""
        df = pl.DataFrame(
            {
                "text": ["a", "b"],
                "is_retweet": [False, False],
                "language": ["pt", "en"],
            }
        )
        result = filter_by_raw_metadata(df)
        assert result["text"].to_list() == ["a"]

    def test_can_disable_both_filters(self) -> None:
        """Com ambos os filtros desligados, nenhuma linha deve ser removida."""
        df = pl.DataFrame(
            {
                "text": ["a", "b"],
                "is_retweet": [True, False],
                "language": ["en", "pt"],
            }
        )
        result = filter_by_raw_metadata(df, exclude_retweets=False, required_language=None)
        assert result.height == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocessing.py::TestFilterByRawMetadata -v`
Expected: FAIL with `ImportError: cannot import name 'filter_by_raw_metadata' from 'preprocessing.filtering'`

- [ ] **Step 3: Write minimal implementation**

In `src/preprocessing/filtering.py`, add the import and the new function. Update the import block at the top:

```python
import logging

import polars as pl

from constants.columns import IS_RETWEET_COLUMN, LANGUAGE_COLUMN
from preprocessing.cleaning import (
    is_minimum_length_content,
    is_probable_portuguese_text,
    is_spam_like,
)

logger = logging.getLogger(__name__)
```

Add the new function after `filter_by_inclusion_criteria` (end of file):

```python
def filter_by_raw_metadata(
    dataframe: pl.DataFrame,
    *,
    exclude_retweets: bool = True,
    required_language: str | None = "pt",
) -> pl.DataFrame:
    """Remove retweets e/ou linhas fora do idioma alvo, usando metadados da coleta.

    Aplicado antes da normalização por linha (mais barato: usa colunas já
    tipadas da coleta, não exige inspecionar o texto). Replies
    (``is_reply``) não são afetadas por este filtro — ainda expressam
    opinião original do autor, diferente de um retweet (texto duplicado
    de outro usuário).

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame de entrada, contendo ao menos ``is_retweet`` (bool) e,
        quando ``required_language`` não for ``None``, ``language`` (str).
    exclude_retweets : bool, optional
        Se ``True``, remove linhas com ``is_retweet=True``, by default True.
    required_language : str | None, optional
        Idioma exigido (comparação exata com a coluna ``language``); ``None``
        desliga o filtro de idioma, by default "pt".

    Returns
    -------
    pl.DataFrame
        Subconjunto do DataFrame original que atende aos critérios habilitados.

    Examples
    --------
    >>> df = pl.DataFrame(
    ...     {"text": ["a", "b"], "is_retweet": [True, False], "language": ["pt", "pt"]}
    ... )
    >>> filter_by_raw_metadata(df)["text"].to_list()
    ['b']
    """
    filtered = dataframe
    if exclude_retweets:
        before = filtered.height
        filtered = filtered.filter(~pl.col(IS_RETWEET_COLUMN))
        logger.info("Filtro de retweet: %d/%d linha(s) mantida(s)", filtered.height, before)
    if required_language is not None:
        before = filtered.height
        filtered = filtered.filter(pl.col(LANGUAGE_COLUMN) == required_language)
        logger.info(
            "Filtro de idioma ('%s'): %d/%d linha(s) mantida(s)",
            required_language,
            filtered.height,
            before,
        )
    return filtered
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/filtering.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): adicionar filter_by_raw_metadata (retweet/idioma)

Filtra por metadados da coleta (is_retweet, language) antes da
normalização por linha, mais barato que a heurística lexical existente."
```

---

## Task 6: Parallelize `preprocessing/pipeline.py::run_preprocessing_pipeline`

Depends on Task 1 (exception pickling fix): the parallel path re-raises `PipelineStageError` from inside a `ProcessPoolExecutor` worker.

**Files:**
- Modify: `src/parallel/preprocessing.py` (generalize type hints only, no behavior change)
- Modify: `src/preprocessing/pipeline.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `run_parallel_text_cleaning` (existing, generalized), `parallel.core.ParallelExecutionResult`.
- Produces: `run_preprocessing_pipeline(..., max_workers: int | None = None, show_progress: bool = True) -> pl.DataFrame` — same return contract, two new keyword-only params. Task 7 consumes this.

Note on the `parallel/preprocessing.py` type-hint generalization: `run_parallel_text_cleaning` delegates straight through to the already-fully-generic `execute_parallel_tasks`, so passing `tuple[int, str]` items works at runtime today regardless of its declared `Callable[[str], str]`/`Iterable[str]` annotations — Python does not enforce type hints. There is no runtime-observable red state for that specific change, so it is folded into Step 3 below as a type-correctness cleanup (needed so `basedpyright` in CI doesn't flag the indexed-tuple call added to `preprocessing/pipeline.py`), not given its own red/green cycle. The real, genuinely-failing behavior this task adds is `run_preprocessing_pipeline` accepting `max_workers`/`show_progress` and keeping the normalized column aligned to its original row when run with multiple workers — that is what Steps 1-2 below test.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_preprocessing.py`, inside `class TestRunPreprocessingPipeline:` (after `test_raises_pipeline_stage_error_when_normalization_fails`):

```python
def test_normalization_stays_aligned_to_original_rows_under_parallelism(self) -> None:
    """A normalização paralela não deve embaralhar a correspondência texto->linha.

    Com vários workers, a coleta de resultados termina na ordem de
    conclusão, não na ordem de submissão (ver ``parallel/core.py``) —
    sem reordenar pelo índice original, a coluna ``text_normalized``
    ficaria associada à linha errada.
    """
    n_rows = 20
    df = pl.DataFrame(
        {
            "id": [str(index) for index in range(n_rows)],
            "text": [
                f"RT @user{index}: mensagem numero {index} muito boa" for index in range(n_rows)
            ],
        }
    )

    result = run_preprocessing_pipeline(
        df, apply_inclusion_filters=False, max_workers=4, show_progress=False
    )

    for index in range(n_rows):
        row = result.filter(pl.col("id") == str(index))
        assert f"numero {index}" in row["text_normalized"].to_list()[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocessing.py::TestRunPreprocessingPipeline::test_normalization_stays_aligned_to_original_rows_under_parallelism -v`
Expected: FAIL with `TypeError: run_preprocessing_pipeline() got an unexpected keyword argument 'max_workers'`

- [ ] **Step 3: Write minimal implementation**

In `src/parallel/preprocessing.py`, generalize the type hints (pure typing change, no behavior change):

```python
"""Paralelização de etapas de pré-processamento de texto.

Usa múltiplos processos (``ProcessPoolExecutor``) para distribuir a limpeza
e normalização de textos entre os núcleos disponíveis, já que essas
operações (regex, tokenização, remoção de acentos — ver
``src/preprocessing/``) são tipicamente ligadas a CPU, não a I/O.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

ItemType = TypeVar("ItemType")
ResultType = TypeVar("ResultType")


def run_parallel_text_cleaning(
    clean_text_func: Callable[[ItemType], ResultType],
    texts: Iterable[ItemType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função de limpeza/normalização a múltiplos itens em paralelo.

    Distribui o processamento entre múltiplos processos, adequado para
    operações ligadas a CPU como remoção de acentos, normalização de
    espaços e aplicação de expressões regulares. Aceita tanto textos
    simples quanto itens indexados (``tuple[int, str]``), usados quando o
    chamador precisa realinhar os resultados à ordem original de entrada
    (ver ``src/preprocessing/pipeline.py``).

    Parameters
    ----------
    clean_text_func : Callable[[ItemType], ResultType]
        Função de limpeza aplicada a cada item. Deve ser importável no
        nível de módulo (não local nem lambda), pois é serializada para os
        processos filhos.
    texts : Iterable[ItemType]
        Itens a serem limpos (textos simples, ou pares indexados).
    max_workers : int | None, optional
        Número máximo de processos usados, by default None (o executor
        escolhe automaticamente com base nos núcleos disponíveis).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[ItemType, ResultType]
        Itens limpos com sucesso e falhas isoladas por item, cada uma
        preservando o item original que causou o erro.

    Examples
    --------
    >>> resultado = run_parallel_text_cleaning(str.strip, ["  a  ", " b "])  # doctest: +SKIP
    >>> sorted(resultado.successes)  # doctest: +SKIP
    ['a', 'b']
    """
    return execute_parallel_tasks(
        clean_text_func,
        texts,
        executor_class=ProcessPoolExecutor,
        max_workers=max_workers,
        task_description="Limpeza paralela de texto",
        show_progress=show_progress,
    )
```

Now edit `src/preprocessing/pipeline.py`. Update the imports at the top:

```python
import functools
import logging
import operator

import polars as pl

from exceptions.pipeline import PipelineStageError
from parallel.preprocessing import run_parallel_text_cleaning
from preprocessing.cleaning import clean_tweet_text
from preprocessing.emojis import normalize_emojis
from preprocessing.filtering import filter_by_inclusion_criteria
from preprocessing.text import (
    normalize_hashtags,
    normalize_mentions,
    normalize_repeated_characters,
    normalize_urls,
)
from preprocessing.tokenization import tokenize_and_normalize
from utils.text import normalize_whitespace
from utils.validation import validate_not_empty_collection

logger = logging.getLogger(__name__)

_IndexedText = tuple[int, str]
```

Keep `normalize_tweet_text` and `_normalize_row_text` exactly as they are. Add a new function right after `_normalize_row_text` (before `run_preprocessing_pipeline`):

```python
def _normalize_indexed_row_text(item: _IndexedText, *, keep_hashtag_word: bool) -> _IndexedText:
    """Normaliza um texto indexado, preservando sua posição original na lista de entrada.

    Necessário porque a execução paralela (``ProcessPoolExecutor``) coleta
    resultados na ordem de conclusão, não na ordem de submissão — sem o
    índice original, a coluna resultante ficaria desalinhada em relação às
    demais linhas do DataFrame.

    Parameters
    ----------
    item : tuple[int, str]
        Par ``(índice_original, texto_bruto)``.
    keep_hashtag_word : bool
        Repassado a :func:`_normalize_row_text`.

    Returns
    -------
    tuple[int, str]
        Par ``(índice_original, texto_normalizado)``.
    """
    original_index, text = item
    return original_index, _normalize_row_text(text, keep_hashtag_word=keep_hashtag_word)
```

Now replace `run_preprocessing_pipeline`'s signature and the normalization step in its body. Change the signature line:

```python
def run_preprocessing_pipeline(
    dataframe: pl.DataFrame,
    *,
    text_column: str = "text",
    normalized_text_column: str = "text_normalized",
    tokens_column: str | None = None,
    keep_hashtag_word: bool = True,
    expand_slang: bool = True,
    apply_negation_marking: bool = True,
    apply_inclusion_filters: bool = True,
    minimum_characters: int = 5,
    minimum_words: int = 2,
    minimum_portuguese_ratio: float = 0.15,
    max_repeated_word_ratio: float = 0.5,
    drop_duplicate_text: bool = True,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
```

Add to the docstring `Parameters` section (after `keep_hashtag_word`'s entry):

```
    max_workers : int | None, optional
        Número máximo de processos usados na normalização paralela, by
        default None (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.
```

Replace the body's normalization block (currently the list comprehension building `normalized_texts`) with:

```python
    validate_not_empty_collection(dataframe, collection_name="dataframe")

    indexed_texts = list(enumerate(dataframe[text_column].to_list()))
    normalization_result = run_parallel_text_cleaning(
        functools.partial(_normalize_indexed_row_text, keep_hashtag_word=keep_hashtag_word),
        indexed_texts,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    if normalization_result.failures:
        raise normalization_result.failures[0].error
    normalized_texts = [
        text for _, text in sorted(normalization_result.successes, key=operator.itemgetter(0))
    ]
    result = dataframe.with_columns(pl.Series(normalized_text_column, normalized_texts))
```

(Everything below this — the `apply_inclusion_filters`/`tokens_column` blocks and the final `logger.info`/`return` — stays exactly as it is.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (all tests, including the existing `test_raises_pipeline_stage_error_when_normalization_fails`, which now exercises the fail-fast path through a `ProcessPoolExecutor` worker — this only works because of Task 1's fix)

- [ ] **Step 5: Commit**

```bash
git add src/parallel/preprocessing.py src/preprocessing/pipeline.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): paralelizar a normalização de texto por linha

run_preprocessing_pipeline normaliza os textos em ProcessPoolExecutor
via run_parallel_text_cleaning, usando o padrão de item indexado +
reordenação por índice (mesmo de inference/llm_batch.py) para manter a
coluna resultante alinhada às linhas originais. Fail-fast preservado."
```

---

## Task 7: Wire `pipelines/preprocessing.py::run_preprocessing_stage` to the batch loader and raw-metadata filter

**Files:**
- Modify: `src/pipelines/preprocessing.py`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `load_raw_tweet_batch` (Task 4), `filter_by_raw_metadata` (Task 5), `run_preprocessing_pipeline(..., max_workers, show_progress)` (Task 6).
- Produces: `run_preprocessing_stage(paths, *, max_workers=None, show_progress=True, exclude_retweets=True, required_language="pt", **preprocessing_overrides) -> Path` — same return contract (`paths.normalized_corpus_file`), no longer reads `paths.raw_tweets_file`.

- [ ] **Step 1: Write the failing test**

In `tests/test_pipelines.py`, add `from datetime import datetime` to the imports at the top. Replace `class TestRunPreprocessingStage:`'s existing test with:

```python
class TestRunPreprocessingStage:
    """Testes de :func:`pipelines.preprocessing.run_preprocessing_stage`."""

    def test_loads_raw_batch_and_writes_normalized_corpus(
        self, pipeline_paths: ProjectPaths
    ) -> None:
        """Deve carregar o lote bruto de data/raw/ e gravar o corpus normalizado correspondente."""
        raw_batch = pl.DataFrame(
            {
                "tweet_id": ["1", "2"],
                "user_id": ["u1", "u1"],
                "text": ["RT @a: muito bom!! 😍", "RT @b: péssimo produto"],
                "created_at": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
                "language": ["pt", "pt"],
                "is_reply": [False, False],
                "is_retweet": [False, False],
                "like_count": [0, 0],
                "reply_count": [0, 0],
                "retweet_count": [0, 0],
                "quote_count": [0, 0],
                "source_query": [None, None],
                "source_group": [None, None],
            }
        )
        write_dataset(raw_batch, pipeline_paths.data_raw_dir / "usuario_teste.parquet")

        normalized_path = run_preprocessing_stage(
            pipeline_paths, show_progress=False, apply_inclusion_filters=False
        )

        assert normalized_path == pipeline_paths.normalized_corpus_file
        normalized_corpus = read_dataset_file(normalized_path)
        assert normalized_corpus.height == 2
        assert "text_normalized" in normalized_corpus.columns

    def test_excludes_retweets_before_normalization(self, pipeline_paths: ProjectPaths) -> None:
        """Um tweet marcado como retweet deve ser removido antes da normalização."""
        raw_batch = pl.DataFrame(
            {
                "tweet_id": ["1", "2"],
                "user_id": ["u1", "u1"],
                "text": ["muito bom o produto", "RT texto duplicado"],
                "created_at": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
                "language": ["pt", "pt"],
                "is_reply": [False, False],
                "is_retweet": [False, True],
                "like_count": [0, 0],
                "reply_count": [0, 0],
                "retweet_count": [0, 0],
                "quote_count": [0, 0],
                "source_query": [None, None],
                "source_group": [None, None],
            }
        )
        write_dataset(raw_batch, pipeline_paths.data_raw_dir / "usuario_teste.parquet")

        run_preprocessing_stage(pipeline_paths, show_progress=False, apply_inclusion_filters=False)

        normalized_corpus = read_dataset_file(pipeline_paths.normalized_corpus_file)
        assert normalized_corpus.height == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipelines.py::TestRunPreprocessingStage -v`
Expected: FAIL — `run_preprocessing_stage` still calls `load_raw_tweet_dataset(paths.raw_tweets_file)`, which no longer exists (removed in Task 4), so this raises `ImportError` at collection time or `AttributeError`/`DataNotFoundError` at call time.

- [ ] **Step 3: Write minimal implementation**

Replace the full content of `src/pipelines/preprocessing.py`:

```python
"""Normalização e limpeza do corpus bruto de tweets.

Implementa o estágio ``preprocessing`` de ``configs/config.yaml -> stages``:
carrega em paralelo o lote de tweets brutos coletados por usuário
(``data/raw/*.parquet`` — ver ``src/data/loader.py``), filtra por metadados
da coleta (retweet/idioma — ``src/preprocessing/filtering.py``), aplica o
pipeline de normalização/limpeza de ``src/preprocessing/pipeline.py`` e
grava o resultado no corpus normalizado (``paths.normalized_corpus_file``).
"""

import logging
from pathlib import Path
from typing import Any

from config.paths import ProjectPaths
from data.loader import load_raw_tweet_batch
from data.writer import write_dataset
from preprocessing.filtering import filter_by_raw_metadata
from preprocessing.pipeline import run_preprocessing_pipeline

logger = logging.getLogger(__name__)


def run_preprocessing_stage(
    paths: ProjectPaths,
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
    exclude_retweets: bool = True,
    required_language: str | None = "pt",
    **preprocessing_overrides: Any,
) -> Path:
    """Executa a etapa de pré-processamento sobre o lote de tweets brutos.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto (``configs/paths.yaml``).
    max_workers : int | None, optional
        Repassado ao carregamento em lote
        (:func:`data.loader.load_raw_tweet_batch`) e à normalização
        paralela (:func:`preprocessing.pipeline.run_preprocessing_pipeline`),
        by default None (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe barras de progresso no console, by default True.
    exclude_retweets : bool, optional
        Repassado a :func:`preprocessing.filtering.filter_by_raw_metadata`,
        by default True.
    required_language : str | None, optional
        Repassado a :func:`preprocessing.filtering.filter_by_raw_metadata`,
        by default "pt".
    **preprocessing_overrides : Any
        Hiperparâmetros repassados a
        :func:`preprocessing.pipeline.run_preprocessing_pipeline` (ex.:
        ``apply_inclusion_filters``, ``tokens_column``).

    Returns
    -------
    Path
        Caminho do corpus normalizado escrito (``paths.normalized_corpus_file``).

    Raises
    ------
    EmptyDatasetError
        Se ``data/raw/`` não contiver arquivos, ou se o corpus ficar vazio
        após os filtros.
    PipelineStageError
        Se a normalização de algum texto do corpus falhar.

    Examples
    --------
    >>> run_preprocessing_stage(paths)  # doctest: +SKIP
    """
    raw_batch = load_raw_tweet_batch(
        paths.data_raw_dir, max_workers=max_workers, show_progress=show_progress
    )
    filtered_batch = filter_by_raw_metadata(
        raw_batch, exclude_retweets=exclude_retweets, required_language=required_language
    )
    normalized_corpus = run_preprocessing_pipeline(
        filtered_batch,
        max_workers=max_workers,
        show_progress=show_progress,
        **preprocessing_overrides,
    )
    write_dataset(normalized_corpus, paths.normalized_corpus_file)

    logger.info(
        "Etapa de pré-processamento concluída: %d/%d linha(s) mantida(s) (de %d brutas).",
        normalized_corpus.height,
        filtered_batch.height,
        raw_batch.height,
    )
    return paths.normalized_corpus_file
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipelines.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/preprocessing.py tests/test_pipelines.py
git commit -m "feat(pipelines): preprocessing lê o lote de data/raw/ diretamente

run_preprocessing_stage carrega e concatena em paralelo os *.parquet de
data/raw/ (um por usuário), filtra retweets/idioma antes de normalizar,
e não depende mais de um arquivo único consolidado por ingestion."
```

---

## Task 8: `parallel/labeling.py` — parallel sentiment labeling wrapper

**Files:**
- Create: `src/parallel/labeling.py`
- Test: `tests/test_parallel.py`

**Interfaces:**
- Consumes: `parallel.core.execute_parallel_tasks`/`ParallelExecutionResult` (existing).
- Produces: `run_parallel_sentiment_labeling(label_func: Callable[[ItemType], ResultType], items: Iterable[ItemType], *, max_workers: int | None = None, show_progress: bool = True) -> ParallelExecutionResult[ItemType, ResultType]`. Task 9 consumes this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parallel.py`: import `run_parallel_sentiment_labeling` alongside the other `parallel.*` imports, add a module-level helper near the other test helpers:

```python
def _label_or_fail(text: str) -> str:
    """Retorna o texto em maiúsculas, ou levanta ValueError para o texto 'erro'."""
    if text == "erro":
        raise ValueError("texto inválido")
    return text.upper()
```

Then add:

```python
class TestRunParallelSentimentLabeling:
    """Testes da paralelização de rotulagem de sentimento (``parallel.labeling``)."""

    def test_labels_all_items_successfully(self) -> None:
        """Todos os itens devem ser rotulados com sucesso quando não há erro."""
        result = run_parallel_sentiment_labeling(
            str.upper, ["a", "b"], show_progress=False, max_workers=1
        )
        assert sorted(result.successes) == ["A", "B"]
        assert result.failures == []

    def test_isolates_failure_per_item(self) -> None:
        """A falha de rotulagem de um item não deve interromper os demais."""
        result = run_parallel_sentiment_labeling(
            _label_or_fail, ["ok", "erro"], show_progress=False, max_workers=1
        )
        assert result.successes == ["OK"]
        assert len(result.failures) == 1
        assert result.failures[0].item == "erro"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parallel.py::TestRunParallelSentimentLabeling -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'parallel.labeling'`

- [ ] **Step 3: Write minimal implementation**

Create `src/parallel/labeling.py`:

```python
"""Paralelização da rotulagem automática de sentimento.

Usa múltiplos processos (``ProcessPoolExecutor``), adequado para o
rotulador heurístico-lexical (``src/labeling/automatic.py``): regex e
contagem léxica são operações ligadas a CPU, não a I/O.
"""

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from parallel.core import ParallelExecutionResult, execute_parallel_tasks

ItemType = TypeVar("ItemType")
ResultType = TypeVar("ResultType")


def run_parallel_sentiment_labeling(
    label_func: Callable[[ItemType], ResultType],
    items: Iterable[ItemType],
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> ParallelExecutionResult[ItemType, ResultType]:
    """Aplica uma função de rotulagem de sentimento a múltiplos itens em paralelo.

    Parameters
    ----------
    label_func : Callable[[ItemType], ResultType]
        Função de rotulagem aplicada a cada item. Deve ser importável no
        nível de módulo (não local nem lambda), pois é serializada para os
        processos filhos.
    items : Iterable[ItemType]
        Itens a serem rotulados (textos simples, ou pares indexados — ver
        ``src/labeling/automatic.py``).
    max_workers : int | None, optional
        Número máximo de processos usados, by default None (o executor
        escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.

    Returns
    -------
    ParallelExecutionResult[ItemType, ResultType]
        Itens rotulados com sucesso e falhas isoladas por item, cada uma
        preservando o item original que causou o erro.

    Examples
    --------
    >>> resultado = run_parallel_sentiment_labeling(str.upper, ["a", "b"])  # doctest: +SKIP
    """
    return execute_parallel_tasks(
        label_func,
        items,
        executor_class=ProcessPoolExecutor,
        max_workers=max_workers,
        task_description="Rotulagem paralela de sentimento",
        show_progress=show_progress,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parallel.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/parallel/labeling.py tests/test_parallel.py
git commit -m "feat(parallel): adicionar rotulagem paralela de sentimento

Nova run_parallel_sentiment_labeling (ProcessPoolExecutor), usada pela
cascata de rotuladores em labeling/automatic.py."
```

---

## Task 9: Parallelize `labeling/automatic.py::run_cascade_labeling`

**Files:**
- Modify: `src/labeling/automatic.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Consumes: `run_parallel_sentiment_labeling` (Task 8).
- Produces: `run_cascade_labeling(..., max_workers: int | None = None, show_progress: bool = True) -> pl.DataFrame` — same return contract (long-format DataFrame validated against `LabelingResultSchema`), two new keyword-only params. Task 10 consumes this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_labeling.py`, inside `class TestRunCascadeLabeling:` (after `test_raises_data_validation_error_for_invalid_labeler_output`):

```python
def test_produces_same_result_with_and_without_parallelism(self) -> None:
    """A rotulagem paralela deve produzir exatamente o mesmo resultado que a execução padrão.

    Verifica em particular que a ordem/correspondência amostra->rótulo
    não se perde ao coletar resultados em ProcessPoolExecutor (ver
    _label_indexed_item / operator.itemgetter(0)).
    """
    df = pl.DataFrame(
        {
            "id": [str(i) for i in range(6)],
            "text": [
                "adorei o produto",
                "péssimo atendimento",
                "chegou no prazo",
                "excelente experiência",
                "produto horrível",
                "sem opinião formada",
            ],
        }
    )
    labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}

    result_sequential = run_cascade_labeling(df, labelers, max_workers=1, show_progress=False)
    result_parallel = run_cascade_labeling(df, labelers, max_workers=4, show_progress=False)

    assert result_sequential.sort("id").to_dicts() == result_parallel.sort("id").to_dicts()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_labeling.py::TestRunCascadeLabeling::test_produces_same_result_with_and_without_parallelism -v`
Expected: FAIL with `TypeError: run_cascade_labeling() got an unexpected keyword argument 'max_workers'`

- [ ] **Step 3: Write minimal implementation**

In `src/labeling/automatic.py`, update the imports at the top:

```python
import functools
import logging
import operator
import re
from collections.abc import Mapping
from typing import Protocol

import polars as pl

from constants.labels import NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL
from parallel.labeling import run_parallel_sentiment_labeling
from preprocessing.emojis import calculate_emoji_sentiment_counts
from schemas.labeling import validate_labeling_result
from utils.validation import validate_not_empty_collection
```

Keep everything from `logger = logging.getLogger(__name__)` through the end of the `SentimentLabeler` protocol class exactly as-is. Add, right after the `SentimentLabeler` protocol and before `calculate_lexicon_sentiment_counts`:

```python
_IndexedText = tuple[int, str]
_IndexedLabel = tuple[int, str, float]


def _label_indexed_item(item: _IndexedText, *, labeler: SentimentLabeler) -> _IndexedLabel:
    """Classifica um texto indexado, preservando sua posição original na lista de entrada.

    Necessário porque a execução paralela (``ProcessPoolExecutor``) coleta
    resultados na ordem de conclusão, não na ordem de submissão.

    Parameters
    ----------
    item : tuple[int, str]
        Par ``(índice_original, texto)``.
    labeler : SentimentLabeler
        Rotulador aplicado ao texto.

    Returns
    -------
    tuple[int, str, float]
        Tripla ``(índice_original, rótulo_de_sentimento, confiança)``.
    """
    original_index, text = item
    sentiment_label, confidence_score = labeler.label(text)
    return original_index, sentiment_label, confidence_score
```

Keep `calculate_lexicon_sentiment_counts`, `classify_by_lexical_heuristic` and `LexicalHeuristicLabeler` exactly as they are. Replace `run_cascade_labeling`'s signature and body:

```python
def run_cascade_labeling(
    dataframe: pl.DataFrame,
    labelers: Mapping[str, SentimentLabeler],
    *,
    id_column: str = "id",
    text_column: str = "text",
    weights: Mapping[str, float] | None = None,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Executa a cascata de rotuladores sobre um corpus, produzindo candidatos por amostra.

    Cada combinação (amostra, rotulador) gera uma linha no formato longo
    exigido por :class:`schemas.labeling.LabelingResultSchema`, insumo de
    ``src/labeling/consensus.py`` e ``src/labeling/confidence.py``. Para
    cada rotulador, todos os textos são classificados em paralelo (ver
    :func:`parallel.labeling.run_parallel_sentiment_labeling`), com os
    resultados reordenados pelo índice original antes de montar as
    colunas — a coleta paralela retorna na ordem de conclusão, não na
    ordem de submissão.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Corpus de entrada, contendo ao menos ``id_column`` e
        ``text_column``. Não vazio.
    labelers : Mapping[str, SentimentLabeler]
        Rotuladores a executar, nomeados pela chave (ex.:
        ``"heuristica_lexica"``, correspondendo a ``cascade.labelers`` de
        ``configs/labeling.yaml``). Não vazio.
    id_column : str, optional
        Nome da coluna identificadora de cada amostra, by default "id".
    text_column : str, optional
        Nome da coluna de texto a ser classificada, by default "text".
    weights : Mapping[str, float] | None, optional
        Peso de cada rotulador pelo nome usado em ``labelers``, repassado a
        ``src/labeling/consensus.py`` na agregação ponderada. Rotuladores
        ausentes do mapeamento recebem peso 1.0, by default None.
    max_workers : int | None, optional
        Número máximo de processos usados por rotulador, by default None
        (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console por
        rotulador, by default True.

    Returns
    -------
    pl.DataFrame
        DataFrame no formato longo (``id``, ``tagger``, ``sentiment_label``,
        ``confidence_score``, ``weight``), validado contra
        :class:`schemas.labeling.LabelingResultSchema`.

    Raises
    ------
    EmptyDatasetError
        Se ``dataframe`` ou ``labelers`` estiverem vazios.
    DataValidationError
        Se algum resultado produzido violar o contrato de dados (ex.:
        rótulo fora de :data:`constants.labels.SENTIMENT_CLASSES`).

    Examples
    --------
    >>> df = pl.DataFrame({"id": ["1"], "text": ["adorei o produto"]})
    >>> labelers = {"heuristica_lexica": LexicalHeuristicLabeler()}
    >>> resultado = run_cascade_labeling(
    ...     df, labelers, weights={"heuristica_lexica": 1.0}, show_progress=False
    ... )
    >>> resultado["sentiment_label"].to_list()
    ['positivo']
    """
    validate_not_empty_collection(dataframe, collection_name="dataframe")
    validate_not_empty_collection(labelers, collection_name="labelers")
    resolved_weights = weights or {}

    row_ids = dataframe[id_column].to_list()
    indexed_texts = list(enumerate(dataframe[text_column].to_list()))

    ids: list[str] = []
    taggers: list[str] = []
    sentiment_labels: list[str] = []
    confidences: list[float] = []
    label_weights: list[float] = []

    for tagger_name, labeler in labelers.items():
        labeling_result = run_parallel_sentiment_labeling(
            functools.partial(_label_indexed_item, labeler=labeler),
            indexed_texts,
            max_workers=max_workers,
            show_progress=show_progress,
        )
        if labeling_result.failures:
            raise labeling_result.failures[0].error
        for original_index, sentiment_label, confidence_score in sorted(
            labeling_result.successes, key=operator.itemgetter(0)
        ):
            ids.append(row_ids[original_index])
            taggers.append(tagger_name)
            sentiment_labels.append(sentiment_label)
            confidences.append(confidence_score)
            label_weights.append(resolved_weights.get(tagger_name, 1.0))

    result = pl.DataFrame(
        {
            "id": ids,
            "tagger": taggers,
            "sentiment_label": sentiment_labels,
            "confidence_score": confidences,
            "weight": label_weights,
        }
    )
    logger.info(
        "Rotulagem em cascata concluída: %d amostra(s) x %d rotulador(es) = %d resultado(s).",
        dataframe.height,
        len(labelers),
        result.height,
    )
    return validate_labeling_result(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_labeling.py -v`
Expected: PASS (all tests in the file, including the existing `test_raises_data_validation_error_for_invalid_labeler_output`, which requires `_FakeInvalidLabeler` — a module-level class in the test file — to be picklable for `ProcessPoolExecutor`; it is, since it has no non-picklable state)

- [ ] **Step 5: Commit**

```bash
git add src/labeling/automatic.py tests/test_labeling.py
git commit -m "feat(labeling): paralelizar a cascata de rotuladores por texto

run_cascade_labeling classifica todos os textos de cada rotulador em
ProcessPoolExecutor via run_parallel_sentiment_labeling, com o mesmo
padrão de item indexado + reordenação por índice da etapa de
preprocessing. Fail-fast preservado (nenhuma linha de rotulagem
silenciosamente descartada)."
```

---

## Task 10: Wire `max_workers`/`show_progress` into `pipelines/labeling.py`; document the no-parallelism decision in `pipelines/features.py`

**Files:**
- Modify: `src/pipelines/labeling.py`
- Modify: `src/pipelines/features.py` (docstring only)

**Interfaces:**
- Consumes: `run_cascade_labeling(..., max_workers, show_progress)` (Task 9).
- Produces: `run_labeling_stage(..., max_workers: int | None = None, show_progress: bool = True) -> Path` — same return contract, two new keyword-only params.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipelines.py`, inside `class TestRunLabelingStage:` (after the two existing tests):

```python
def test_accepts_max_workers_and_show_progress(self, pipeline_paths: ProjectPaths) -> None:
    """Deve aceitar e repassar max_workers/show_progress sem alterar o resultado."""
    normalized_corpus = pl.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["adorei o produto", "produto pessimo"],
            "text_normalized": ["adorei o produto", "produto pessimo"],
        }
    )
    write_dataset(normalized_corpus, pipeline_paths.normalized_corpus_file)

    labeled_path = run_labeling_stage(
        pipeline_paths,
        {"heuristica_lexica": LexicalHeuristicLabeler()},
        max_workers=2,
        show_progress=False,
    )

    labeled_corpus = read_dataset_file(labeled_path)
    assert labeled_corpus.sort("id")["sentiment_label"].to_list() == ["positivo", "negativo"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipelines.py::TestRunLabelingStage::test_accepts_max_workers_and_show_progress -v`
Expected: FAIL with `TypeError: run_labeling_stage() got an unexpected keyword argument 'max_workers'`

- [ ] **Step 3: Write minimal implementation**

In `src/pipelines/labeling.py`, update `run_labeling_stage`'s signature (add two keyword-only parameters after `minimum_kappa`):

```python
def run_labeling_stage(
    paths: ProjectPaths,
    labelers: Mapping[str, SentimentLabeler],
    *,
    text_column: str = "text_normalized",
    weights: Mapping[str, float] | None = None,
    select_for_human_validation: bool = True,
    human_validation_sample_size: int = 500,
    human_validation_labels: pl.DataFrame | None = None,
    gold_set: pl.DataFrame | None = None,
    minimum_kappa: float = 0.6,
    max_workers: int | None = None,
    show_progress: bool = True,
) -> Path:
```

Add to the docstring `Parameters` section (after `minimum_kappa`'s entry):

```
    max_workers : int | None, optional
        Repassado a :func:`labeling.automatic.run_cascade_labeling`, by
        default None (o executor escolhe automaticamente).
    show_progress : bool, optional
        Se ``True``, exibe uma barra de progresso no console, by default
        True.
```

Update the call to `run_cascade_labeling` inside the function body:

```python
    labeling_results = run_cascade_labeling(
        normalized_corpus,
        labelers,
        text_column=text_column,
        weights=weights,
        max_workers=max_workers,
        show_progress=show_progress,
    )
```

(Everything else in the function stays unchanged.)

In `src/pipelines/features.py`, add a paragraph to `run_features_stage`'s docstring, right after the summary line (`"""Executa o split estratificado do corpus rotulado e a extração de features TF-IDF.`) and before the blank line that precedes `Parameters`:

```python
def run_features_stage(
    paths: ProjectPaths,
    *,
    label_column: str = "sentiment_label",
    text_column: str = "text",
    tfidf_overrides: dict[str, Any] | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> FeatureArtifacts:
    """Executa o split estratificado do corpus rotulado e a extração de features TF-IDF.

    Esta etapa não paraleliza o cálculo do TF-IDF: ele já é vetorizado em
    C via ``scikit-learn`` (rápido mesmo para grandes lotes) e não lê
    ``data/raw`` em nenhum momento — o ganho de paralelismo para grandes
    volumes de tweets de usuários acontece a montante, no carregamento do
    lote bruto e na normalização/rotulagem por linha (ver
    ``pipelines.preprocessing.run_preprocessing_stage`` e
    ``pipelines.labeling.run_labeling_stage``). Envolver a chamada ao
    ``scikit-learn`` em um ``ProcessPoolExecutor`` só adicionaria overhead
    de serialização sem ganho real.

    Parameters
    ----------
```

(The rest of the docstring and the entire function body stay unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipelines.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/labeling.py src/pipelines/features.py tests/test_pipelines.py
git commit -m "feat(pipelines): repassar max_workers/show_progress à etapa labeling

Também documenta explicitamente por que run_features_stage não recebe
paralelismo próprio (TF-IDF já vetorizado, sem leitura de data/raw)."
```

---

## Task 11: Wire `--max-workers` CLI flag into preprocessing/labeling stages; update `parallel/__init__.py` exports

**Files:**
- Modify: `src/main.py`
- Modify: `src/parallel/__init__.py`

**Interfaces:**
- Consumes: `run_preprocessing_stage(..., max_workers)` (Task 7), `run_labeling_stage(..., max_workers)` (Task 10), `run_parallel_parquet_loading` (Task 3), `run_parallel_sentiment_labeling` (Task 8).
- Produces: no new public interface; wires the existing `--max-workers` CLI flag through to two more stages, and re-exports the two new `parallel.*` functions from the package `__init__`, matching the existing convention where every symbol in a `parallel/*.py` module is re-exported from `parallel/__init__.py`.

- [ ] **Step 1: Update `src/main.py`**

Update the `--max-workers` argument's help text (find `parser.add_argument("--max-workers", ...)`):

```python
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Número máximo de threads/processos paralelos (etapas "
            "`ingestion`/`preprocessing`/`labeling`/`llm_evaluation`)."
        ),
    )
```

Update `_build_preprocessing_stage_kwargs`:

```python
def _build_preprocessing_stage_kwargs(
    paths: ProjectPaths, general_config: GeneralConfig, settings: Settings, args: argparse.Namespace
) -> dict[str, Any]:
    """Monta os argumentos de :func:`pipelines.preprocessing.run_preprocessing_stage`.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos resolvidos do projeto.
    general_config : GeneralConfig
        Configuração geral validada, não utilizada diretamente nesta etapa.
    settings : Settings
        Configurações sensíveis ao ambiente, não utilizadas diretamente
        nesta etapa.
    args : argparse.Namespace
        Argumentos de linha de comando (``--max-workers``).

    Returns
    -------
    dict[str, Any]
        Argumentos nomeados para :func:`pipelines.preprocessing.run_preprocessing_stage`.
    """
    del general_config, settings
    return {"paths": paths, "max_workers": args.max_workers}
```

Update `_build_labeling_stage_kwargs`'s docstring `args` entry and its `return` statement (keep everything else in the function identical):

```python
    args : argparse.Namespace
        Argumentos de linha de comando (``--max-workers``).
```

```python
    return {
        "paths": paths,
        "labelers": labelers,
        "weights": weights,
        "human_validation_sample_size": labeling_config["human_validation"]["sample_size"],
        "minimum_kappa": labeling_config["validation"]["minimum_agreement"],
        "max_workers": args.max_workers,
    }
```

- [ ] **Step 2: Sanity-check `main.py` still parses and dispatches correctly**

Run: `uv run python -m src.main --help`
Expected: prints the CLI help text without any traceback, showing the updated `--max-workers` help text mentioning `preprocessing`/`labeling`.

(`src/main.py` is excluded from coverage — `pyproject.toml` `[tool.coverage.run] omit` — so no new automated test is required here, consistent with the project's existing choice to test stage orchestration only through the stage functions themselves, already covered by Tasks 7 and 10.)

- [ ] **Step 3: Update `src/parallel/__init__.py`**

Replace the full content:

```python
"""Execução paralela e concorrente de etapas do pipeline.

Camada de utilitários genéricos de paralelismo (``concurrent.futures``),
usada pelos módulos de pré-processamento, rotulagem, inferência,
experimentos, coleta de dados e carregamento em lote para distribuir
trabalho entre processos (tarefas ligadas a CPU) ou threads (tarefas
ligadas a I/O), isolando a falha de um item sem interromper o restante do
lote.

Modules
-------
core
    Motor genérico de execução paralela (``execute_parallel_tasks``) e os
    tipos de resultado (``ParallelExecutionResult``, ``ParallelTaskFailure``)
    compartilhados pelos demais módulos.
data_loading
    Leitura paralela de lotes de arquivos Parquet.
experiments
    Execução paralela de múltiplos experimentos/configurações de treino.
inference
    Execução paralela de inferência/predição de modelos.
labeling
    Execução paralela da rotulagem automática de sentimento.
preprocessing
    Execução paralela de limpeza e normalização de texto.
scraping
    Execução paralela de coleta de dados (scraping).
"""

from parallel.core import ParallelExecutionResult, ParallelTaskFailure, execute_parallel_tasks
from parallel.data_loading import run_parallel_parquet_loading
from parallel.experiments import run_parallel_experiments
from parallel.inference import run_parallel_predictions
from parallel.labeling import run_parallel_sentiment_labeling
from parallel.preprocessing import run_parallel_text_cleaning
from parallel.scraping import run_parallel_scraping

__all__: list[str] = [
    "ParallelExecutionResult",
    "ParallelTaskFailure",
    "execute_parallel_tasks",
    "run_parallel_experiments",
    "run_parallel_parquet_loading",
    "run_parallel_predictions",
    "run_parallel_scraping",
    "run_parallel_sentiment_labeling",
    "run_parallel_text_cleaning",
]
```

- [ ] **Step 4: Run the full test suite once more to confirm nothing broke**

Run: `uv run pytest tests/test_parallel.py tests/test_pipelines.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/main.py src/parallel/__init__.py
git commit -m "feat(main): repassar --max-workers às etapas preprocessing/labeling

Também atualiza parallel/__init__.py para reexportar
run_parallel_parquet_loading e run_parallel_sentiment_labeling,
mantendo a convenção existente de reexportar todo símbolo público de
cada submódulo de parallel/."
```

---

## Task 12: Full regression pass

**Files:** none (verification only — fix forward in the relevant file from Tasks 1-11 if something regresses)

- [ ] **Step 1: Run the full test suite with coverage**

Run: `uv run pytest -m "not slow"`
Expected: all tests PASS; the terminal coverage summary (`--cov-report=term-missing`, configured in `pyproject.toml`) shows overall coverage ≥ 80%. If any test fails or coverage drops below 80%, fix the specific file/test from the task that introduced the regression before proceeding — do not weaken the 80% gate.

- [ ] **Step 2: Run the slow-marked tests too, if any exist for the touched modules**

Run: `uv run pytest -m slow`
Expected: PASS, or "no tests ran" if none are marked `slow` in the touched files.

- [ ] **Step 3: Lint and format check**

Run: `uv run ruff check src tests`
Expected: no errors. Fix any reported issue (import order, unused import, line length) directly in the affected file from Tasks 1-11.

Run: `uv run ruff format --check src tests`
Expected: no files would be reformatted. If any would, run `uv run ruff format src tests` and re-run the full test suite (Step 1) to confirm formatting didn't change behavior.

- [ ] **Step 4: Manual end-to-end smoke check against the real `data/raw/` files**

Run:
```bash
uv run python -c "
from config.paths import load_project_paths
from data.loader import load_raw_tweet_batch

paths = load_project_paths()
batch = load_raw_tweet_batch(paths.data_raw_dir, show_progress=False)
print('linhas:', batch.height)
print('colunas:', batch.columns)
print('is_retweet=True:', batch.filter(batch['is_retweet']).height)
print('language != pt:', batch.filter(batch['language'] != 'pt').height)
"
```
Expected: runs without error, prints `linhas: 9800`, `colunas` including `id` (not `tweet_id`), and non-zero counts for both filters (confirming the retweet/language filters in Task 5 have real rows to act on against the actual `data/raw/` corpus).

- [ ] **Step 5: Commit only if Steps 1-4 required fixes**

If no fixes were needed, there is nothing to commit for this task — it is a pure verification checkpoint. If fixes were required, commit them with a message describing what regressed and why, e.g.:

```bash
git add <arquivo(s) corrigido(s)>
git commit -m "fix: corrigir regressão encontrada na verificação final

<descrição específica do que quebrou e por quê>"
```
