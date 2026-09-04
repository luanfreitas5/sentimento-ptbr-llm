"""Utilitários genéricos do HypotheSAEs: prompts, truncamento e cache em JSON.

Funções de apoio usadas por praticamente todos os outros módulos de
``hypothesaes`` (``annotate``, ``interpret_neurons``, ``evaluation``): carga
e cache de templates de prompt (``src/hypothesaes/prompts/*.txt``),
truncamento de texto por palavras/caracteres/tokens antes de enviá-lo a um
LLM, filtragem de textos inválidos e leitura/escrita simples de JSON.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from exceptions.data import DataNotFoundError

logger = logging.getLogger(__name__)

PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

_PROMPT_TEMPLATE_CACHE: dict[str, str] = {}


def load_prompt_template(prompt_name: str) -> str:
    """Carrega um template de prompt de ``src/hypothesaes/prompts/`` (com cache em memória).

    Parameters
    ----------
    prompt_name : str
        Nome do template, sem a extensão ``.txt`` (ex.: ``"annotate"``).

    Returns
    -------
    str
        Conteúdo bruto do template, pronto para ``str.format(**kwargs)``.

    Raises
    ------
    DataNotFoundError
        Se não existir um arquivo ``{prompt_name}.txt`` em ``PROMPTS_DIR``.

    Examples
    --------
    >>> "{hypothesis}" in load_prompt_template("annotate")
    True
    """
    if prompt_name in _PROMPT_TEMPLATE_CACHE:
        return _PROMPT_TEMPLATE_CACHE[prompt_name]

    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_path.is_file():
        raise DataNotFoundError(str(prompt_path))

    content = prompt_path.read_text(encoding="utf-8")
    _PROMPT_TEMPLATE_CACHE[prompt_name] = content
    return content


def _truncate_by_words(text: str, max_words: int | None) -> str:
    """Trunca ``text`` para no máximo ``max_words`` palavras, se aplicável."""
    if max_words is None:
        return text
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def _truncate_by_chars(text: str, max_chars: int | None) -> str:
    """Trunca ``text`` para no máximo ``max_chars`` caracteres, se aplicável."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _truncate_by_tokens(text: str, max_tokens: int | None) -> str:
    """Trunca ``text`` para no máximo ``max_tokens`` tokens (encoding ``cl100k_base``),
    se aplicável.
    """
    if max_tokens is None:
        return text

    import tiktoken  # type: ignore[reportMissingImports]

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else text


def truncate_text(
    text: str,
    max_words: int | None = None,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    truncation_message: str = "[... restante do texto foi truncado]",
) -> str:
    """Trunca um texto por número de palavras, caracteres e/ou tokens.

    As três condições são aplicadas em sequência (palavras, depois
    caracteres, depois tokens), cada uma operando sobre o resultado da
    anterior, de forma que o limite mais restritivo prevalece.

    Parameters
    ----------
    text : str
        Texto de entrada a truncar.
    max_words : int | None, optional
        Número máximo de palavras (separadas por espaço), by default None.
    max_chars : int | None, optional
        Número máximo de caracteres, by default None.
    max_tokens : int | None, optional
        Número máximo de tokens, contados com o encoding ``cl100k_base``
        (``tiktoken``), by default None.
    truncation_message : str, optional
        Sufixo anexado ao texto truncado, indicando que houve corte, by
        default "[... restante do texto foi truncado]".

    Returns
    -------
    str
        Texto truncado (com ``truncation_message`` anexado) ou o texto
        original, caso nenhum limite seja informado ou nenhum seja
        excedido.

    Examples
    --------
    >>> truncate_text("um dois tres quatro", max_words=2)
    'um dois[... restante do texto foi truncado]'
    >>> truncate_text("texto curto", max_words=10)
    'texto curto'
    """
    if all(limit is None for limit in (max_words, max_chars, max_tokens)):
        return text

    if text.endswith(truncation_message):
        return text

    truncated = _truncate_by_words(text, max_words)
    truncated = _truncate_by_chars(truncated, max_chars)
    truncated = _truncate_by_tokens(truncated, max_tokens)

    if truncated != text:
        truncated += truncation_message

    return truncated


def format_text_for_display(text: str, max_chars: int = 128) -> str:
    """Trunca um texto e remove quebras de linha, para exibição compacta em logs/prints.

    Parameters
    ----------
    text : str
        Texto de entrada.
    max_chars : int, optional
        Número máximo de caracteres exibidos, by default 128.

    Returns
    -------
    str
        Texto truncado, com quebras de linha substituídas por espaço.

    Examples
    --------
    >>> format_text_for_display("linha um\\nlinha dois", max_chars=50)
    'linha um linha dois'
    """
    return truncate_text(text, max_chars=max_chars).replace("\n", " ")


def filter_invalid_texts(texts: Sequence[str | None]) -> list[str]:
    """Remove valores ``None`` e strings vazias/apenas-espaço de uma lista de textos.

    Parameters
    ----------
    texts : Sequence[str | None]
        Lista de textos, potencialmente contendo ``None`` ou strings vazias.

    Returns
    -------
    list[str]
        Lista filtrada, preservando a ordem original.

    Examples
    --------
    >>> filter_invalid_texts(["ok", None, "  ", "outro"])
    ['ok', 'outro']
    """
    original_count = len(texts)
    filtered_texts = [text for text in texts if text is not None and text.strip()]
    discarded_count = original_count - len(filtered_texts)

    if discarded_count > 0:
        logger.warning(
            "Ignorando %d item(ns) None ou string(s) vazia(s) de um total de %d.",
            discarded_count,
            original_count,
        )

    return filtered_texts


def save_json(data: dict[str, Any], file_path: Path) -> None:
    """Salva um dicionário em um arquivo JSON, criando diretórios ausentes.

    Parameters
    ----------
    data : dict[str, Any]
        Dados a serem serializados.
    file_path : Path
        Caminho do arquivo de destino.

    Examples
    --------
    >>> import tempfile
    >>> destino = Path(tempfile.mkdtemp()) / "cache" / "exemplo.json"
    >>> save_json({"a": 1}, destino)
    >>> destino.is_file()
    True
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data), encoding="utf-8")


def load_json(file_path: Path) -> dict[str, Any]:
    """Carrega um dicionário de um arquivo JSON, retornando vazio se o arquivo não existir.

    Parameters
    ----------
    file_path : Path
        Caminho do arquivo de origem.

    Returns
    -------
    dict[str, Any]
        Conteúdo desserializado, ou ``{}`` se ``file_path`` não existir.

    Examples
    --------
    >>> load_json(Path("caminho/inexistente.json"))
    {}
    """
    if not file_path.is_file():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))
