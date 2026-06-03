"""
Prompt utilizado para classificação de phishing.

O mesmo prompt é usado para todos os modelos (Claude e os modelos locais) para garantir uma comparação justa.

Formato de saída esperado (JSON):
    {
      "classification": "phishing" | "legitimate",
      "phishing_likelihood": <inteiro de 0 a 100>,
      "reasons": ["<motivo curto>", "<motivo curto>", ...]
    }
"""
import re
import json

SYSTEM_PROMPT = """You are given the full content of an email. Your task is to decide whether the
email is a phishing attempt or a legitimate message.

Analyze the email for common phishing indicators, such as:
- Sender information that is unusual or inconsistent with the organization it
  claims to come from (for example, a display name or address that does not
  match the expected domain);
- Suspicious links or attachments, such as shortened URLs, links whose visible
  text does not match their real destination, or unexpected attachments;
- Urgent, threatening, or pressuring language meant to force a hasty action;
- Requests for sensitive information, such as passwords, financial data, or
  personal identification.

Then make a decision, estimate how likely the email is to be a phishing attempt
on a scale from 0 to 100, and give a brief justification.

Respond ONLY with a single JSON object, with no extra text, in exactly this
format:
{
  "classification": "phishing" or "legitimate",
  "phishing_likelihood": <integer from 0 to 100>,
  "reasons": ["<short reason>", "<short reason>", ...]
}"""

USER_PROMPT_TEMPLATE = """Email to analyse:

Subject: {subject}

Body:
{body}"""


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


def build_prompt(subject: str, body: str) -> tuple[str, str]:
    """Retorna (system_prompt, user_prompt) para o e-mail fornecido.
    Trunca o corpo do e-mail para 2000 caracteres se for muito longo, para evitar exceder limites de token.
    Args:
        subject: Assunto do e-mail
        body: Corpo do e-mail
    Returns:
        Tuple com system_prompt e user_prompt formatados
    """
    body_truncated = body[:2000] if len(body) > 2000 else body
    user_prompt = USER_PROMPT_TEMPLATE.format(
        subject=subject,
        body=body_truncated,
    )
    return SYSTEM_PROMPT, user_prompt


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

    return None


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

    # classification → rótulo interno
    classification = data.get("classification")
    predicted = None
    if isinstance(classification, str) and classification.strip().lower() in VALID_CLASSIFICATIONS:
        predicted = _CLASSIFICATION_MAP[classification.strip().lower()]
    else:
        notes.append(f"classification inválida: {classification!r}")

    # phishing_likelihood: inteiro 0..100 (bool é subclasse de int → rejeitado)
    likelihood = data.get("phishing_likelihood")
    if isinstance(likelihood, bool) or not isinstance(likelihood, int) or not (0 <= likelihood <= 100):
        notes.append(f"phishing_likelihood inválido: {likelihood!r}")
        likelihood = None

    # reasons: lista não-vazia de strings
    reasons = data.get("reasons")
    if isinstance(reasons, list) and reasons and all(isinstance(r, str) for r in reasons):
        reasons = [r.strip() for r in reasons]
    else:
        notes.append(f"reasons inválida: {reasons!r}")
        reasons = []

    return {
        "predicted":           predicted,
        "phishing_likelihood": likelihood,
        "reasons":             reasons,
        "parse_note":          "; ".join(notes),
    }
