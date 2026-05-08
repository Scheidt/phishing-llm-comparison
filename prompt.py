"""
Prompt utilizado para classificação de phishing.

O mesmo prompt é usado para todos os modelos (Claude e os modelos locais) para garantir uma comparação justa.
"""

SYSTEM_PROMPT = """Você é um especialista em cibersegurança com foco em detecção de phishing.
Sua tarefa é analisar e-mails e classificá-los como PHISHING ou LEGÍTIMO.

Regras:
- Responda APENAS com uma palavra: PHISHING ou LEGÍTIMO
- Não adicione explicações, pontuação ou qualquer outro texto
- Seja objetivo e preciso"""

USER_PROMPT_TEMPLATE = """Analise o seguinte e-mail e classifique:

Assunto: {subject}

Corpo:
{body}

Classificação:"""


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


def parse_response(raw_response: str) -> str | None:
    """
    Extrai a classificação da resposta bruta do modelo.
    Retorna 'PHISHING', 'LEGÍTIMO' ou None (resposta não reconhecida).
    """
    cleaned = raw_response.strip().upper()

    if any(tok in cleaned for tok in ["PHISHING", "PHISH"]):
        return "PHISHING"
    if any(tok in cleaned for tok in ["LEGÍTIMO", "LEGITIMO", "LEGITIMATE", "LEGIT"]):
        return "LEGÍTIMO"
    
    print (f"WARNING! Resposta não reconhecida: '{raw_response}' → '{cleaned}'")
    return None