"""Proteção de dados pessoais (LGPD) para textos de tweets e identificadores de autores.

Concentra três responsabilidades de fronteira, usadas antes de qualquer dado sair
da camada ``data/``:

* **Pseudonimização** de identificadores com HMAC-SHA256 e chave secreta vinda do
  ``.env`` (nunca de argumento de linha de comando, que aparece no histórico do shell);
* **Mascaramento** de @menções, URLs, e-mails, CPFs válidos e telefones em textos que
  irão para relatórios, figuras, tabelas ou planilhas de anotação;
* **Contagem** de dados pessoais por categoria, para auditar saídas sem imprimir o
  conteúdo encontrado.

O mascaramento é heurístico (expressões regulares): reduz a exposição, mas não é uma
garantia de anonimização. Nomes próprios e outros quase-identificadores no texto livre
não são detectados.
"""

import hashlib
import hmac
import os
import re
from collections.abc import Iterable

import polars as pl

from exceptions.configuration import MissingEnvironmentVariableError

PSEUDONYMIZATION_KEY_ENV_VAR = "PSEUDONYMIZATION_KEY"
PSEUDONYM_LENGTH = 20

MENTION_PLACEHOLDER = "@usuario"
URL_PLACEHOLDER = "<url>"
EMAIL_PLACEHOLDER = "<email>"
CPF_PLACEHOLDER = "<cpf>"
PHONE_PLACEHOLDER = "<telefone>"

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+")
_CPF_PATTERN = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?55[\s-]?)?\(?\d{2}\)?[\s-]?9?\d{4}[\s-]?\d{4}(?!\d)")
_MENTION_PATTERN = re.compile(r"(?<![\w@])@\w+")

PERSONAL_DATA_CATEGORIES: tuple[str, ...] = ("mencao", "url", "email", "cpf", "telefone")

# Colunas de texto livre de tweets que podem ir para relatórios e planilhas.
TEXT_COLUMN_NAMES: frozenset[str] = frozenset(
    {"text", "text_normalized", "texto", "texto_normalizado"}
)
TEXT_COLUMN_PREFIXES: tuple[str, ...] = ("top_example", "exemplo", "example")


def resolve_pseudonymization_key(env_var: str = PSEUDONYMIZATION_KEY_ENV_VAR) -> str:
    """Lê a chave de pseudonimização da variável de ambiente.

    Parameters
    ----------
    env_var : str, optional
        Nome da variável, by default ``"PSEUDONYMIZATION_KEY"``.

    Returns
    -------
    str
        Chave secreta.

    Raises
    ------
    MissingEnvironmentVariableError
        Se a variável não estiver definida ou estiver vazia.

    Examples
    --------
    >>> resolve_pseudonymization_key("VARIAVEL_INEXISTENTE")  # doctest: +SKIP
    """
    key = os.environ.get(env_var, "")
    if not key:
        raise MissingEnvironmentVariableError(env_var)
    return key


def pseudonymize_value(value: str, key: str) -> str:
    """Pseudonimiza um identificador com HMAC-SHA256 (irreversível sem a chave).

    Diferente de um hash simples, o HMAC com chave secreta não pode ser
    revertido por força bruta sobre o espaço de identificadores sem a chave.

    Parameters
    ----------
    value : str
        Identificador original (por exemplo, o id do autor do tweet).
    key : str
        Chave secreta de pseudonimização.

    Returns
    -------
    str
        Os primeiros :data:`PSEUDONYM_LENGTH` caracteres hexadecimais do HMAC.

    Examples
    --------
    >>> pseudonymize_value("123", "chave") == pseudonymize_value("123", "chave")
    True
    >>> pseudonymize_value("123", "a") == pseudonymize_value("123", "b")
    False
    """
    digest = hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:PSEUDONYM_LENGTH]


def validate_cpf(value: str) -> bool:
    """Indica se o texto é um CPF com dígitos verificadores válidos.

    Parameters
    ----------
    value : str
        Texto com 11 dígitos, com ou sem pontuação.

    Returns
    -------
    bool
        ``True`` se os dois dígitos verificadores conferirem e os dígitos não forem
        todos iguais.

    Examples
    --------
    >>> validate_cpf("529.982.247-25")
    True
    >>> validate_cpf("111.111.111-11")
    False
    """
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for position in (9, 10):
        weighted_sum = sum(
            digit * weight
            for digit, weight in zip(digits, range(position + 1, 1, -1), strict=False)
        )
        expected = (weighted_sum * 10) % 11 % 10
        if digits[position] != expected:
            return False
    return True


def mask_personal_data(text: str, *, mask_mentions: bool = True) -> str:
    """Substitui dados pessoais detectáveis por marcadores genéricos.

    A ordem importa: e-mails antes das @menções (o e-mail contém ``@``) e URLs antes
    de números (URLs podem conter sequências de dígitos).

    Parameters
    ----------
    text : str
        Texto de entrada.
    mask_mentions : bool, optional
        Se ``True``, troca @menções por ``@usuario``, by default True.

    Returns
    -------
    str
        Texto com e-mails, URLs, CPFs válidos, telefones e @menções mascarados.

    Examples
    --------
    >>> mask_personal_data("fala @fulano veja https://exemplo.com ou ligue (11) 91234-5678")
    'fala @usuario veja <url> ou ligue <telefone>'
    """
    masked = _EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, text)
    masked = _URL_PATTERN.sub(URL_PLACEHOLDER, masked)
    masked = _CPF_PATTERN.sub(
        lambda match: CPF_PLACEHOLDER if validate_cpf(match.group()) else match.group(), masked
    )
    masked = _PHONE_PATTERN.sub(PHONE_PLACEHOLDER, masked)
    if mask_mentions:
        masked = _MENTION_PATTERN.sub(MENTION_PLACEHOLDER, masked)
    return masked


def count_personal_data(texts: Iterable[str]) -> dict[str, int]:
    """Conta quantos textos contêm cada categoria de dado pessoal.

    Não devolve nem registra o conteúdo encontrado, apenas contagens, para que a
    auditoria não reproduza o dado pessoal que procura.

    Parameters
    ----------
    texts : Iterable[str]
        Textos a inspecionar.

    Returns
    -------
    dict[str, int]
        Número de textos com ao menos uma ocorrência de cada categoria em
        :data:`PERSONAL_DATA_CATEGORIES`, mais a chave ``"total"`` com o número de textos
        inspecionados.

    Examples
    --------
    >>> count_personal_data(["oi @alguem", "sem nada"])["mencao"]
    1
    """
    counts = dict.fromkeys(PERSONAL_DATA_CATEGORIES, 0)
    total = 0
    for text in texts:
        total += 1
        # o próprio marcador "@usuario" não conta como menção real
        if any(
            match.group() != MENTION_PLACEHOLDER
            for match in _MENTION_PATTERN.finditer(_EMAIL_PATTERN.sub("", text))
        ):
            counts["mencao"] += 1
        if _URL_PATTERN.search(text):
            counts["url"] += 1
        if _EMAIL_PATTERN.search(text):
            counts["email"] += 1
        if any(validate_cpf(match.group()) for match in _CPF_PATTERN.finditer(text)):
            counts["cpf"] += 1
        if _PHONE_PATTERN.search(_URL_PATTERN.sub("", text)):
            counts["telefone"] += 1
    counts["total"] = total
    return counts


def mask_text_columns(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Mascara dados pessoais nas colunas de texto livre de um DataFrame de relatório.

    Atua só nas colunas de texto de tweet (nomes em :data:`TEXT_COLUMN_NAMES` ou com
    prefixo em :data:`TEXT_COLUMN_PREFIXES`); as demais passam intactas.

    Parameters
    ----------
    dataframe : pl.DataFrame
        DataFrame que será gravado em relatório, figura ou planilha.

    Returns
    -------
    pl.DataFrame
        Cópia com as colunas de texto mascaradas por :func:`mask_personal_data`.

    Examples
    --------
    >>> df = pl.DataFrame({"text": ["oi @fulano"], "n": [1]})
    >>> mask_text_columns(df)["text"].to_list()
    ['oi @usuario']
    """
    text_columns = [
        name
        for name, dtype in dataframe.schema.items()
        if dtype == pl.String
        and (name in TEXT_COLUMN_NAMES or name.startswith(TEXT_COLUMN_PREFIXES))
    ]
    if not text_columns:
        return dataframe
    return dataframe.with_columns(
        pl.col(name).map_elements(mask_personal_data, return_dtype=pl.String)
        for name in text_columns
    )
