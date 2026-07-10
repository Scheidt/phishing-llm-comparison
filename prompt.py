"""
Prompt utilizado para classificação de phishing.
"""
import config

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

Estimate how likely the email is to be a phishing attempt
on a scale from 0 to 100, and give a brief justification, then make a decision.

Respond ONLY with a single JSON object, with no extra text, in exactly this
format:
{
  "indicators": ["<short indicator>", "<short indicator>", ...],
  "phishing_likelihood": <integer from 0 to 100>,
  "classification": "phishing" or "legitimate"
}"""

USER_PROMPT_TEMPLATE = """Email to analyse:
From: {from}

Subject: {subject}

Body:
{body}"""


def build_prompt(subject: str, body: str, sender: str = "") -> tuple[str, str]:
    """Retorna (system_prompt, user_prompt) para o e-mail fornecido.
    Se config.TRUNCATE_EMAIL_BODY for True, trunca o corpo do e-mail em
    config.EMAIL_BODY_MAX_CHARS caracteres, para evitar exceder limites de token.
    Args:
        subject: Assunto do e-mail
        body: Corpo do e-mail
        sender: Remetente do e-mail (campo "From")
    Returns:
        Tuple com system_prompt e user_prompt formatados
    """
    if config.TRUNCATE_EMAIL_BODY:
        body = body[:config.EMAIL_BODY_MAX_CHARS]
    # format_map (e não format) porque "from" é palavra reservada do Python e
    # não pode ser passada como argumento nomeado.
    user_prompt = USER_PROMPT_TEMPLATE.format_map({
        "from": sender,
        "subject": subject,
        "body": body,
    })
    return SYSTEM_PROMPT, user_prompt
