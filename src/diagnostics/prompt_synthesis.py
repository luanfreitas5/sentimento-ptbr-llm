"""Geração de ``prompts/v2.md`` a partir das hipóteses validadas.

O v2 é o prompt v1 **intacto** mais um bloco de regras curtas e verificáveis
(ex.: ``ironia elogiosa -> negativo``), inserido logo antes do marcador
``{{TEXTO}}``. As regras são propostas por um LLM a partir das hipóteses que
sobreviveram ao Bonferroni e DEVEM ser revisadas por uma pessoa antes do uso:
uma hipótese sobre discordância descreve *onde* os modelos divergem, não *qual*
é o rótulo correto.

Proveniência (hipóteses, regras, hashes do v1/v2 e modelo) vai para
``<v2>.meta.json``, e não para o prompt, para não poluir o que é enviado ao LLM.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config.paths import resolve_project_path
from diagnostics.llm_client import AsyncLLMClient, DiskCompletionCache
from diagnostics.settings import DiagnosticsSettings
from exceptions.data import DataValidationError
from io_utils.json import write_json
from utils.hashing import calculate_text_hash

logger = logging.getLogger(__name__)

TEXT_PLACEHOLDER = "{{TEXTO}}"
RULES_HEADER = (
    "Regras adicionais (derivadas de hipóteses validadas; aplique-as somente quando o "
    "tweet se encaixar claramente na condição):"
)
_ARROW_PATTERN = re.compile(r"\s*(?:->|→)\s*")
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
RULES_SYSTEM_PROMPT = (
    "Você é especialista em análise de sentimento de tweets em português do Brasil."
)


def build_rules_request(hypotheses: Sequence[str], *, max_rules: int) -> str:
    """Monta o pedido ao LLM para converter hipóteses em regras curtas.

    Parameters
    ----------
    hypotheses : Sequence[str]
        Hipóteses validadas (Bonferroni).
    max_rules : int
        Máximo de regras.

    Returns
    -------
    str
        Prompt em pt-BR.

    Examples
    --------
    >>> "ironia" in build_rules_request(["ironia elogiosa"], max_rules=3)
    True
    """
    listed = "\n".join(f"- {hypothesis}" for hypothesis in hypotheses)
    return (
        "Abaixo há padrões em que modelos de sentimento divergem ou erram ao classificar "
        "tweets (positivo, neutro, negativo).\n\n"
        f"Padrões:\n{listed}\n\n"
        f"Escreva no máximo {max_rules} regras CURTAS e VERIFICÁVEIS, uma por linha, no "
        "formato `condição -> classe` (classe: positivo, neutro ou negativo). Só escreva "
        "uma regra quando o padrão implicar claramente um rótulo; não invente padrões "
        "novos. Responda apenas com as regras."
    )


def parse_rules(completion: str, *, max_rules: int) -> list[str]:
    """Extrai regras ``condição -> classe`` da resposta do LLM.

    Parameters
    ----------
    completion : str
        Resposta bruta.
    max_rules : int
        Máximo de regras a manter.

    Returns
    -------
    list[str]
        Regras únicas, normalizadas para ``condição -> classe``.

    Examples
    --------
    >>> parse_rules("1. ironia elogiosa → negativo\\n- texto vago", max_rules=5)
    ['ironia elogiosa -> negativo']
    """
    rules: list[str] = []
    for line in completion.splitlines():
        cleaned = _BULLET_PATTERN.sub("", line).strip().strip("`")
        parts = _ARROW_PATTERN.split(cleaned)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            continue
        rule = f"{parts[0].strip()} -> {parts[1].strip().rstrip('.')}"
        if rule not in rules:
            rules.append(rule)
    return rules[:max_rules]


def insert_rules_into_prompt(v1_prompt: str, rules: Sequence[str]) -> str:
    """Insere o bloco de regras antes da linha que contém ``{{TEXTO}}``, sem alterar o resto.

    Parameters
    ----------
    v1_prompt : str
        Conteúdo do prompt v1.
    rules : Sequence[str]
        Regras ``condição -> classe``.

    Returns
    -------
    str
        Conteúdo do prompt v2.

    Raises
    ------
    DataValidationError
        Se não houver regras ou o v1 não tiver o marcador ``{{TEXTO}}``.

    Examples
    --------
    >>> insert_rules_into_prompt("Classifique:\\n{{TEXTO}}", ["ironia -> negativo"]).count("ironia")
    1
    """
    if not rules:
        raise DataValidationError(schema_name="PromptV2", detail="nenhuma regra para inserir")
    lines = v1_prompt.splitlines(keepends=True)
    index = next((i for i, line in enumerate(lines) if TEXT_PLACEHOLDER in line), None)
    if index is None:
        raise DataValidationError(
            schema_name="PromptV2", detail=f"o prompt v1 não contém o marcador {TEXT_PLACEHOLDER}"
        )
    block = RULES_HEADER + "\n" + "\n".join(f"- {rule}" for rule in rules) + "\n\n"
    return "".join([*lines[:index], block, *lines[index:]])


async def synthesize_rules_async(
    client: AsyncLLMClient, hypotheses: Sequence[str], *, model: str, max_rules: int
) -> list[str]:
    """Pede ao LLM as regras curtas derivadas das hipóteses.

    Parameters
    ----------
    client : AsyncLLMClient
        Cliente LLM.
    hypotheses : Sequence[str]
        Hipóteses validadas.
    model : str
        Modelo a usar.
    max_rules : int
        Máximo de regras.

    Returns
    -------
    list[str]
        Regras extraídas (pode ser vazia se o modelo não seguir o formato).
    """
    completion = await client.complete(
        build_rules_request(hypotheses, max_rules=max_rules),
        model=model,
        system_prompt=RULES_SYSTEM_PROMPT,
        max_tokens=400,
        namespace="prompt_v2_rules",
    )
    return parse_rules(completion, max_rules=max_rules)


def synthesize_rules(
    settings: DiagnosticsSettings,
    hypotheses: Sequence[str],
    *,
    max_rules: int = 8,
    client: AsyncLLMClient | None = None,
) -> list[str]:
    """Versão síncrona de :func:`synthesize_rules_async` (1 chamada ao LLM, com cache).

    Parameters
    ----------
    settings : DiagnosticsSettings
        Configuração validada.
    hypotheses : Sequence[str]
        Hipóteses validadas.
    max_rules : int, optional
        Máximo de regras, by default 8.
    client : AsyncLLMClient | None, optional
        Cliente pronto (testes), by default None.

    Returns
    -------
    list[str]
        Regras extraídas.
    """

    async def _run() -> list[str]:
        active = client or AsyncLLMClient(
            settings.llm, cache=DiskCompletionCache(resolve_project_path(settings.llm.cache_dir))
        )
        return await synthesize_rules_async(
            active, hypotheses, model=settings.llm.interpreter_model, max_rules=max_rules
        )

    return asyncio.run(_run())


def write_prompt_v2(
    v1_path: Path,
    v2_path: Path,
    rules: Sequence[str],
    *,
    hypotheses: Sequence[str],
    model: str,
    overwrite: bool = False,
) -> Path:
    """Gera o v2 (v1 + regras) e o ``.meta.json`` de proveniência, sem tocar no v1.

    Parameters
    ----------
    v1_path : Path
        Prompt v1 (somente leitura).
    v2_path : Path
        Destino do v2 (ex.: ``prompts/v2.md``).
    rules : Sequence[str]
        Regras revisadas.
    hypotheses : Sequence[str]
        Hipóteses validadas que originaram as regras.
    model : str
        Modelo que propôs as regras.
    overwrite : bool, optional
        Permite sobrescrever um v2 existente, by default False.

    Returns
    -------
    Path
        Caminho do v2 gravado.

    Raises
    ------
    DataValidationError
        Se ``v2_path`` for o próprio v1, ou já existir e ``overwrite`` for falso.

    Examples
    --------
    >>> write_prompt_v2(
    ...     Path("v1.txt"), Path("v2.md"), ["a -> negativo"], hypotheses=["a"], model="m"
    ... )  # doctest: +SKIP
    """
    if v2_path.resolve() == v1_path.resolve():
        raise DataValidationError(schema_name="PromptV2", detail="o v2 não pode sobrescrever o v1")
    if v2_path.exists() and not overwrite:
        raise DataValidationError(
            schema_name="PromptV2", detail=f"{v2_path.name} já existe; use overwrite=True"
        )
    v1_text = v1_path.read_text(encoding="utf-8")
    v2_text = insert_rules_into_prompt(v1_text, rules)
    v2_path.write_text(v2_text, encoding="utf-8")
    metadata: dict[str, Any] = {
        "v1_file": v1_path.name,
        "v1_sha256": calculate_text_hash(v1_text),
        "v2_sha256": calculate_text_hash(v2_text),
        "rules": list(rules),
        "hypotheses": list(hypotheses),
        "rules_model": model,
        "review_required": True,
    }
    write_json(metadata, v2_path.with_suffix(".meta.json"))
    logger.info(
        "Prompt v2 gravado em '%s' com %d regra(s); REVISE antes de usar.", v2_path, len(rules)
    )
    return v2_path
