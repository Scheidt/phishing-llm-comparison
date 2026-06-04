"""
Parsing da resposta dos modelos para a tarefa de classificação de phishing.

Formato de saída esperado (JSON):
    {
      "classification": "phishing" | "legitimate",
      "phishing_likelihood": <inteiro de 0 a 100>,
      "reasons": ["<motivo curto>", "<motivo curto>", ...]
    }
"""
import re
import json


VALID_CLASSIFICATIONS = {"phishing", "legitimate"}

# Mapeia a classificação do modelo (inglês) para o rótulo interno (português).
_CLASSIFICATION_MAP = {
    "phishing":   "PHISHING",
    "legitimate": "LEGÍTIMO",
}

# Remove cercas de código markdown (``` ou ```json) ao redor do JSON.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)
# Captura o primeiro objeto JSON {...} caso o modelo adicione texto ao redor.
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Vírgula pendurada logo antes de um fechamento (inválida em JSON).
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _close_json(text: str) -> str:
    """
    Fecha um objeto JSON truncado/inacabado, acrescentando o que ficou aberto.

    Modelos podem cortar a resposta no meio (ex.: atingiram o limite de tokens)
    e deixar o objeto sem o '}' final, um array sem ']' ou uma string sem as
    aspas de fechamento. Esta função percorre o texto rastreando o que está
    aberto e devolve o texto com os fechamentos que faltam, na ordem correta —
    garantindo que o objeto JSON SEMPRE feche, nem que seja preciso adicionar
    o '}'.

    A função é O(n), mas é o terceiro fallback, e possui latência desprezível
    comparada ao tempo de resposta dos modelos.
    """
    stack = []            # pilha de aberturas '{' / '[' ainda não fechadas
    in_string = False     # estamos dentro de uma string?
    escaped = False       # o caractere anterior foi uma barra de escape?

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    repaired = text
    if in_string:               # fecha string aberta
        repaired += '"'
    for opener in reversed(stack):   # fecha do mais interno ao mais externo
        repaired += "}" if opener == "{" else "]"

    # Uma string/valor truncado pode ter deixado uma vírgula pendurada.
    return _TRAILING_COMMA_RE.sub(r"\1", repaired)


def _extract_json(raw_response: str) -> dict | None:
    """
    Tenta extrair o objeto JSON da resposta bruta do modelo.

    Tolera cercas de código markdown (``` / ```json) e texto extra ao redor
    do JSON. Retorna o dict decodificado ou None se nada válido for encontrado.
    """
    text = raw_response.strip()

    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    # 1ª tentativa: a resposta inteira já é um JSON válido.
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass

    # 2ª tentativa: extrair o primeiro objeto {...} embutido em texto extra.
    match = _OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

    # 3ª tentativa: JSON truncado/inacabado — fecha o que ficou aberto
    # (adiciona o '}' que faltou) e tenta de novo a partir do primeiro '{'.
    start = text.find("{")
    if start != -1:
        try:
            data = json.loads(_close_json(text[start:]))
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _read_classification(data: dict, notes: list) -> str | None:
    """Lê 'classification' e devolve o rótulo interno ('PHISHING'/'LEGÍTIMO') ou None."""
    classification = data.get("classification")

    if not isinstance(classification, str):
        notes.append(f"classification inválida: {classification!r}")
        return None

    chave = classification.strip().lower()
    if chave not in VALID_CLASSIFICATIONS:
        notes.append(f"classification inválida: {classification!r}")
        return None

    return _CLASSIFICATION_MAP[chave]


def _read_likelihood(data: dict, notes: list) -> int | None:
    """Lê 'phishing_likelihood' e devolve um inteiro de 0 a 100, ou None se inválido."""
    likelihood = data.get("phishing_likelihood")

    # bool é subclasse de int, então True/False precisam ser rejeitados à parte.
    eh_inteiro = isinstance(likelihood, int) and not isinstance(likelihood, bool)
    if eh_inteiro and 0 <= likelihood <= 100:
        return likelihood

    notes.append(f"phishing_likelihood inválido: {likelihood!r}")
    return None


def _read_reasons(data: dict, notes: list) -> list:
    """Lê 'reasons' e devolve a lista de motivos (strings), ou lista vazia se inválida."""
    reasons = data.get("reasons")

    eh_lista_de_strings = (
        isinstance(reasons, list)
        and reasons
        and all(isinstance(r, str) for r in reasons)
    )
    if eh_lista_de_strings:
        return [r.strip() for r in reasons]

    notes.append(f"reasons inválida: {reasons!r}")
    return []


def parse_response(raw_response: str) -> dict:
    """
    Faz o parsing da resposta JSON do modelo no formato esperado e extrai
    os campos estruturados.

    Returns:
        Dict com:
            predicted:            'PHISHING' | 'LEGÍTIMO' | None (não reconhecido)
            phishing_likelihood:  int 0..100 | None
            reasons:              list[str] (vazia se ausente/ inválida)
            parse_note:           str descrevendo problemas de parsing (vazio se OK)
    """
    empty = {
        "predicted":           None,
        "phishing_likelihood": None,
        "reasons":             [],
        "parse_note":          "",
    }

    data = _extract_json(raw_response)
    if data is None:
        print(f"WARNING! Resposta não reconhecida (JSON inválido): '{raw_response}'")
        return {**empty, "parse_note": "JSON inválido ou ausente"}

    notes = []

    predicted   = _read_classification(data, notes)
    likelihood  = _read_likelihood(data, notes)
    reasons     = _read_reasons(data, notes)

    return {
        "predicted":           predicted,
        "phishing_likelihood": likelihood,
        "reasons":             reasons,
        "parse_note":          "; ".join(notes),
    }
