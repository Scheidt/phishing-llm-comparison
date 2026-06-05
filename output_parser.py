"""
Parsing da resposta dos modelos para a tarefa de classificação de phishing.

Formato de saída esperado (JSON):
    {
      "classification": "phishing" | "legitimate",
      "phishing_likelihood": <inteiro de 0 a 100>,
      "indicators": ["<motivo curto>", "<motivo curto>", ...]
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


# Caracteres que, logo após uma aspa, indicam que ela FECHA uma string
# (fim de valor: , } ]  /  fim de chave: :). Usados por _escape_inner_quotes.
_STRING_DELIMITERS = {",", "}", "]", ":"}


def _escape_inner_quotes(text: str) -> str:
    """
    Escapa aspas duplas literais que aparecem DENTRO de uma string JSON sem
    estarem escapadas — um erro comum de modelos pequenos, que escrevem algo
    como  "... asks, "what is this?" ..."  deixando as aspas internas cruas e
    quebrando o JSON.

    Heurística: percorre o texto rastreando se estamos dentro de uma string.
    Uma aspa encontrada dentro de uma string só é tratada como FECHAMENTO
    (estrutural) se o próximo caractere não-espaço for um delimitador JSON
    (`, } ] :`) ou o fim do texto; caso contrário é considerada literal e
    recebe uma barra de escape. Aspas já escapadas (\\") são preservadas.

    É uma heurística e pode errar: uma aspa literal seguida logo de ':' (raro)
    seria confundida com o fecho de uma chave. Por isso só roda como último
    recurso, depois que o parsing normal já falhou.
    """
    out = []
    in_string = False
    escaped = False
    n = len(text)

    for i, ch in enumerate(text):
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            continue

        # dentro de uma string
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in _STRING_DELIMITERS:
                out.append(ch)          # aspa estrutural — fecha a string
                in_string = False
            else:
                out.append('\\')        # aspa literal — escapa e segue na string
                out.append(ch)
            continue
        out.append(ch)

    return "".join(out)


def _extract_json(raw_response: str) -> tuple[dict | None, bool]:
    """
    Tenta extrair o objeto JSON da resposta bruta do modelo.

    Tolera cercas de código markdown (``` / ```json) e texto extra ao redor
    do JSON.

    Returns:
        (data, repaired)
        data:     o dict decodificado, ou None se nada válido for encontrado.
        repaired: True se foi preciso REPARAR o conteúdo com heurística
                  (fechar JSON truncado ou escapar aspas internas) para
                  conseguir decodificar. Localizar/extrair um JSON que já era
                  bem-formado (tentativas 1 e 2) NÃO conta como reparo.
    """
    text = raw_response.strip()

    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    # 1ª tentativa: a resposta inteira já é um JSON válido.
    try:
        data = json.loads(text)
        return (data if isinstance(data, dict) else None), False
    except (json.JSONDecodeError, TypeError):
        pass

    # 2ª tentativa: extrair o primeiro objeto {...} embutido em texto extra.
    match = _OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            return (data if isinstance(data, dict) else None), False
        except (json.JSONDecodeError, TypeError):
            pass

    # 3ª tentativa (REPARO): JSON truncado/inacabado — fecha o que ficou aberto
    # (adiciona o '}' que faltou) e tenta de novo a partir do primeiro '{'.
    start = text.find("{")
    if start != -1:
        try:
            data = json.loads(_close_json(text[start:]))
            return (data if isinstance(data, dict) else None), True
        except (json.JSONDecodeError, TypeError):
            pass

    # 4ª tentativa (REPARO): aspas duplas literais não escapadas dentro de
    # strings. Escapa as aspas órfãs e, por garantia, fecha o que sobrar aberto.
    if start != -1:
        try:
            data = json.loads(_close_json(_escape_inner_quotes(text[start:])))
            return (data if isinstance(data, dict) else None), True
        except (json.JSONDecodeError, TypeError):
            pass

    return None, False


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


def _read_indicators(data: dict, notes: list) -> list:
    """Lê 'indicators' e devolve a lista de motivos (strings), ou lista vazia se inválida."""
    indicators = data.get("indicators")

    eh_lista_de_strings = (
        isinstance(indicators, list)
        and indicators
        and all(isinstance(r, str) for r in indicators)
    )
    if eh_lista_de_strings:
        return [r.strip() for r in indicators]

    notes.append(f"indicators inválida: {indicators!r}")
    return []


def parse_response(raw_response: str) -> dict:
    """
    Faz o parsing da resposta JSON do modelo no formato esperado e extrai
    os campos estruturados.

    Returns:
        Dict com:
            predicted:            'PHISHING' | 'LEGÍTIMO' | None (não reconhecido)
            phishing_likelihood:  int 0..100 | None
            indicators:              list[str] (vazia se ausente/ inválida)
            parse_note:           str descrevendo problemas de parsing (vazio se OK)
            repaired:             True se o JSON só pôde ser lido após reparo
                                  heurístico (truncamento ou aspas internas).
    """
    empty = {
        "predicted":           None,
        "phishing_likelihood": None,
        "indicators":             [],
        "parse_note":          "",
        "repaired":            False,
    }

    data, repaired = _extract_json(raw_response)
    if data is None:
        print(f"WARNING! Resposta não reconhecida (JSON inválido): '{raw_response}'")
        return {**empty, "parse_note": "JSON inválido ou ausente"}

    notes = []
    if repaired:
        notes.append("JSON reparado por heurística")

    predicted   = _read_classification(data, notes)
    likelihood  = _read_likelihood(data, notes)
    indicators     = _read_indicators(data, notes)

    return {
        "predicted":           predicted,
        "phishing_likelihood": likelihood,
        "indicators":             indicators,
        "parse_note":          "; ".join(notes),
        "repaired":            repaired,
    }
