"""Aplicação opcional de deploy do projeto de análise de sentimentos pt-BR.

Implementa a Fase 21 do plano de elaboração (ver ``PLANO-ELABORACAO.md``):
uma camada fina de apresentação sobre os artefatos já produzidos pelo
pipeline (``src/pipelines/``), sem nenhuma lógica de treino/avaliação
própria. Habilitação controlada por ``configs/deploy.yaml`` (desabilitada
por padrão — não obrigatória para o escopo acadêmico do projeto).

Modules
-------
api
    API FastAPI para inferência de sentimento sobre texto livre.
dashboard
    Dashboard Streamlit comparativo entre as abordagens do projeto.
"""
