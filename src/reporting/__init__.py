"""Geração de tabelas, Model Cards e Datasheet da etapa ``report``.

Consome as saídas numéricas de ``evaluate`` (``reports/tables/avaliacao/``,
``reports/statistics/``, ``reports/ablation/``) e as converte em artefatos
legíveis e publicáveis, sem nunca incluir texto bruto de tweets (LGPD).

Modules
-------
tables
    Tabelas de resultados com IC 95% em CSV, Markdown e LaTeX
    (:func:`~reporting.tables.build_results_table`,
    :func:`~reporting.tables.dataframe_to_markdown`,
    :func:`~reporting.tables.dataframe_to_latex`).
documents
    Model Cards por categoria de modelo e Datasheet do corpus com os números
    reais (:func:`~reporting.documents.render_model_card`,
    :func:`~reporting.documents.render_datasheet`).
"""
