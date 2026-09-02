"""Utilitários de entrada/saída para diferentes formatos de arquivo.

Camada fina sobre bibliotecas de terceiros (``polars``, ``PyYAML``,
``joblib``), com validação de existência de arquivo e mensagens de erro em
pt-BR.

.. note::
   O pacote chama-se ``io_utils`` (e não ``io``) para não colidir com o
   módulo ``io`` da biblioteca padrão do Python: como ``src/`` é a raiz de
   importação do projeto (``pythonpath = ["src"]``), um pacote chamado
   ``io`` nunca seria alcançável via ``from io.yaml import ...`` — o Python
   sempre resolveria o ``io`` nativo, já em cache em ``sys.modules`` desde a
   inicialização do interpretador.

Modules
-------
csv
    Leitura e escrita de arquivos CSV como DataFrames Polars.
json
    Leitura e escrita de arquivos JSON.
model
    Persistência de modelos treinados via ``joblib``.
parquet
    Leitura e escrita de arquivos Parquet como DataFrames Polars.
yaml
    Leitura e escrita de arquivos YAML.
"""

from io_utils.csv import read_csv, write_csv
from io_utils.json import read_json, write_json
from io_utils.model import load_model, save_model
from io_utils.parquet import read_parquet, write_parquet
from io_utils.yaml import read_yaml, write_yaml

__all__: list[str] = [
    "load_model",
    "read_csv",
    "read_json",
    "read_parquet",
    "read_yaml",
    "save_model",
    "write_csv",
    "write_json",
    "write_parquet",
    "write_yaml",
]
