"""Classes de sentimento e mapeamentos entre rótulo textual e identificador numérico.

Reflete a definição em ``configs/config.yaml`` (``labels.classes``): três
classes de sentimento em português brasileiro, sem atributos sensíveis
associados.
"""

from exceptions.data import DataValidationError

NEGATIVE_LABEL = "negativo"
NEUTRAL_LABEL = "neutro"
POSITIVE_LABEL = "positivo"

# A ordem reflete a ordem usada em ``configs/config.yaml`` e define o
# identificador numérico de cada classe (índice na tupla).
SENTIMENT_CLASSES: tuple[str, ...] = (NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(SENTIMENT_CLASSES)}
ID_TO_LABEL: dict[int, str] = dict(enumerate(SENTIMENT_CLASSES))


def validate_label(label: str) -> str:
    """Valida se um rótulo pertence ao conjunto de classes de sentimento conhecidas.

    Parameters
    ----------
    label : str
        Rótulo textual a ser validado.

    Returns
    -------
    str
        O próprio rótulo, quando válido.

    Raises
    ------
    DataValidationError
        Se o rótulo não pertencer a :data:`SENTIMENT_CLASSES`.

    Examples
    --------
    >>> validate_label("positivo")
    'positivo'
    """
    if label not in SENTIMENT_CLASSES:
        raise DataValidationError(
            schema_name="labels",
            detail=f"rótulo '{label}' não pertence às classes conhecidas {SENTIMENT_CLASSES}",
        )
    return label


def transform_label_to_id(label: str) -> int:
    """Converte um rótulo textual de sentimento em seu identificador numérico.

    Parameters
    ----------
    label : str
        Rótulo textual (ex.: ``"positivo"``).

    Returns
    -------
    int
        Identificador numérico correspondente ao rótulo.

    Raises
    ------
    DataValidationError
        Se o rótulo não pertencer a :data:`SENTIMENT_CLASSES`.

    Examples
    --------
    >>> transform_label_to_id("negativo")
    0
    """
    return LABEL_TO_ID[validate_label(label)]


def transform_id_to_label(label_id: int) -> str:
    """Converte um identificador numérico de sentimento em seu rótulo textual.

    Parameters
    ----------
    label_id : int
        Identificador numérico (índice em :data:`SENTIMENT_CLASSES`).

    Returns
    -------
    str
        Rótulo textual correspondente.

    Raises
    ------
    DataValidationError
        Se o identificador não corresponder a nenhuma classe conhecida.

    Examples
    --------
    >>> transform_id_to_label(2)
    'positivo'
    """
    if label_id not in ID_TO_LABEL:
        raise DataValidationError(
            schema_name="labels",
            detail=f"identificador '{label_id}' não corresponde a nenhuma classe conhecida",
        )
    return ID_TO_LABEL[label_id]
