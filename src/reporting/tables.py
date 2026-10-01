"""Tabelas de resultados formatadas para o relatório (CSV, Markdown e LaTeX).

Converte as tabelas numéricas de ``evaluate`` em tabelas legíveis, com cada
métrica acompanhada do intervalo de confiança (CLAUDE.md, "Rigorous evaluation":
nunca uma métrica pontual isolada).
"""

import polars as pl

from exceptions.data import EmptyDatasetError

_LATEX_ESCAPES: dict[str, str] = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&"}


def format_interval(value: float, low: float | None, high: float | None, digits: int = 3) -> str:
    """Formata ``valor [inferior, superior]``; sem intervalo, só o valor.

    Parameters
    ----------
    value : float
        Valor pontual.
    low : float | None
        Limite inferior do intervalo.
    high : float | None
        Limite superior do intervalo.
    digits : int, optional
        Casas decimais, by default 3.

    Returns
    -------
    str
        Texto formatado.

    Examples
    --------
    >>> format_interval(0.8123, 0.7, 0.9)
    '0.812 [0.700, 0.900]'
    >>> format_interval(0.5, None, None)
    '0.500'
    """
    if low is None or high is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def build_results_table(metrics_table: pl.DataFrame) -> pl.DataFrame:
    """Monta a tabela principal de resultados (uma linha por modelo, métricas com IC 95%).

    Parameters
    ----------
    metrics_table : pl.DataFrame
        Tabela de ``evaluate`` (``reports/tables/avaliacao/metricas_por_modelo.csv``).

    Returns
    -------
    pl.DataFrame
        Colunas ``Categoria``, ``Modelo``, ``n``, ``F1-macro``, ``MCC``, ``Acurácia``, e (quando
        disponíveis) ``ROC-AUC``, ``Brier``, ``ECE`` e ``ms/amostra``.

    Raises
    ------
    EmptyDatasetError
        Se ``metrics_table`` estiver vazia.

    Examples
    --------
    >>> tabela = pl.DataFrame(
    ...     {
    ...         "model": ["svm"],
    ...         "category_label": ["Baseline + ML tradicional"],
    ...         "n_test": [10],
    ...         "f1_macro": [0.7],
    ...         "f1_macro_ci_low": [0.6],
    ...         "f1_macro_ci_high": [0.8],
    ...         "mcc": [0.5],
    ...         "mcc_ci_low": [0.4],
    ...         "mcc_ci_high": [0.6],
    ...         "accuracy": [0.7],
    ...         "accuracy_ci_low": [0.6],
    ...         "accuracy_ci_high": [0.8],
    ...     }
    ... )
    >>> build_results_table(tabela)["F1-macro"].to_list()
    ['0.700 [0.600, 0.800]']
    """
    if metrics_table.is_empty():
        raise EmptyDatasetError("tabela de métricas")

    rows = []
    for row in metrics_table.iter_rows(named=True):
        formatted = {
            "Categoria": row["category_label"],
            "Modelo": row["model"],
            "n": row["n_test"],
        }
        for label, key in (("F1-macro", "f1_macro"), ("MCC", "mcc"), ("Acurácia", "accuracy")):
            formatted[label] = format_interval(
                row[key], row.get(f"{key}_ci_low"), row.get(f"{key}_ci_high")
            )
        optional = (
            ("ROC-AUC", "roc_auc_ovr"),
            ("Brier", "brier_score"),
            ("ECE", "expected_calibration_error"),
            ("ms/amostra", "inference_ms_per_sample"),
        )
        for label, key in optional:
            value = row.get(key)
            formatted[label] = "—" if value is None else f"{value:.3f}"
        rows.append(formatted)
    return pl.DataFrame(rows)


def dataframe_to_markdown(table: pl.DataFrame) -> str:
    """Converte um DataFrame em tabela Markdown.

    Parameters
    ----------
    table : pl.DataFrame
        Tabela a converter.

    Returns
    -------
    str
        Tabela em Markdown (cabeçalho, separador e linhas).

    Examples
    --------
    >>> print(dataframe_to_markdown(pl.DataFrame({"a": [1], "b": ["x"]})))
    | a | b |
    | --- | --- |
    | 1 | x |
    """
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join("—" if cell is None else str(cell) for cell in row) + " |"
        for row in table.iter_rows()
    ]
    return "\n".join([header, separator, *body])


def dataframe_to_latex(table: pl.DataFrame, *, caption: str, label: str) -> str:
    """Converte um DataFrame em tabela LaTeX (``booktabs``), pronta para o artigo.

    Parameters
    ----------
    table : pl.DataFrame
        Tabela a converter.
    caption : str
        Legenda da tabela.
    label : str
        Rótulo ``\\label`` para referência cruzada.

    Returns
    -------
    str
        Ambiente ``table`` com ``tabular`` e regras ``booktabs``.

    Examples
    --------
    >>> "\\\\toprule" in dataframe_to_latex(pl.DataFrame({"a": [1]}), caption="c", label="t")
    True
    """

    def _escape(text: object) -> str:
        """Escapa caracteres especiais do LaTeX em um valor.

        Parameters
        ----------
        text : object
            Valor a escapar.

        Returns
        -------
        str
            Texto seguro para LaTeX.
        """
        return "".join(_LATEX_ESCAPES.get(char, char) for char in str(text))

    column_format = "l" * len(table.columns)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(_escape(column) for column in table.columns) + r" \\",
        r"\midrule",
        *(
            " & ".join("—" if cell is None else _escape(cell) for cell in row) + r" \\"
            for row in table.iter_rows()
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
