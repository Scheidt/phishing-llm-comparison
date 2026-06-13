"""
TEstar consistência de formato de saída (JSON).

Objetivo
--------
Modelos menores podem falhar em retornar a resposta no formato esperado
quando o prompt é grande (e-mail longo + instruções). Este teste roda
o mesmo prompt e modelo várias vezes (definido em REPETITIONS), e verifica
se o modelo consegue retornar de forma consistente um JSON válido no
formato esperado.

Formato JSON esperado
---------------------
    {
      "classification": "phishing" | "legitimate",
      "phishing_likelihood": <inteiro de 0 a 100>,
      "indicators": ["<motivo curto>", "<motivo curto>", ...]
    }

O e-mail (prompt grande) é lido de um arquivo de texto externo
(EMAIL_FILE), sem truncamento, justamente para estressar o cenário de
prompt grande.

Os resultados são gravados em llm_logs/ com "output_testing" no nome do
arquivo, uma linha por repetição.

Uso
---
    python tests/test_output.py

Ajuste os parâmetros no bloco "CONFIGURAÇÃO DO TESTE" abaixo.
"""
import os
import re
import sys
import csv
import json
from datetime import datetime

# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config import LOCAL_MODELS, LLM_LOGS_DIR
from models.dmr_client import DmrClient

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO TESTE  (ajuste conforme necessário)
# ─────────────────────────────────────────────────────────────────────────
MODEL_KEY       = "gemma3_4b_qat"   # chave em config.LOCAL_MODELS
REPETITIONS     = 20                # quantas vezes rodar o mesmo prompt
EMAIL_FILE      = os.path.join(os.path.dirname(__file__), "long_email.txt")
TEST_MAX_TOKENS = 512               # tokens suficientes para o JSON completo

# ─────────────────────────────────────────────────────────────────────────
# PROMPT (dividido entre system e user)
#
# SYSTEM_PROMPT carrega as instruções e o formato de saída; o e-mail entra no
# USER_PROMPT_TEMPLATE via {email}. (Sem .format() no SYSTEM_PROMPT, então as
# chaves do exemplo de JSON ficam literais.)
# ─────────────────────────────────────────────────────────────────────────
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
  "indicators": ["<short indicator>", "<short indicator>", ...]
}"""

USER_PROMPT_TEMPLATE = """Subject: {subject}

Body:
{body}"""

VALID_CLASSIFICATIONS = {"phishing", "legitimate"}


FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def strip_code_fence(text: str) -> tuple[str, bool]:
    """
    Remove cercas de código markdown (``` ou ```json) ao redor do conteúdo.
    Retorna (texto_sem_cerca, tinha_cerca).
    """
    m = FENCE_RE.match(text.strip())
    if m:
        return m.group(1).strip(), True
    return text.strip(), False


# ─────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO DO FORMATO
# ─────────────────────────────────────────────────────────────────────────
def validate_format(raw_response: str) -> dict:
    """
    Verifica se 'raw_response' é um JSON válido no formato esperado.

    Retorna um dict com flags booleanas por checagem e um 'format_ok' geral.
    Cercas de código markdown (``` / ```json) são toleradas: são removidas
    antes do parse e apenas anotadas em 'validation_note'. Texto extra fora
    do JSON, JSON malformado ou campos ausentes/com tipo errado fazem a
    checagem falhar, comportamento que o teste se propõe a detectar.
    """
    result = {
        "is_valid_json":     False,
        "has_classification": False,
        "has_likelihood":     False,
        "has_indicators":        False,
        "format_ok":          False,
        "validation_note":    "",
    }

    text, had_fence = strip_code_fence(raw_response)
    if had_fence:
        fence_note = "envolto em cerca de código; "
    else:
        fence_note = ""

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        result["validation_note"] = f"{fence_note}JSON inválido: {e}"
        return result

    result["is_valid_json"] = True

    if not isinstance(data, dict):
        result["validation_note"] = f"{fence_note}JSON não é um objeto"
        return result

    # classification
    classification = data.get("classification")
    if isinstance(classification, str) and classification.lower() in VALID_CLASSIFICATIONS:
        result["has_classification"] = True

    # phishing_likelihood: inteiro 0..100 (bool é subclasse de int → rejeitado)
    likelihood = data.get("phishing_likelihood")
    if isinstance(likelihood, int) and not isinstance(likelihood, bool) and 0 <= likelihood <= 100:
        result["has_likelihood"] = True

    # indicators: lista não-vazia de strings
    indicators = data.get("indicators")
    if (isinstance(indicators, list) and len(indicators) > 0
            and all(isinstance(r, str) for r in indicators)):
        result["has_indicators"] = True

    result["format_ok"] = (
        result["has_classification"]
        and result["has_likelihood"]
        and result["has_indicators"]
    )

    if not result["format_ok"]:
        checagens = [
            ("classification", result["has_classification"]),
            ("phishing_likelihood", result["has_likelihood"]),
            ("indicators", result["has_indicators"]),
        ]
        faltando = []
        for nome, ok in checagens:
            if not ok:
                faltando.append(nome)
        result["validation_note"] = f"{fence_note}Campos inválidos/ausentes: {', '.join(faltando)}"
    else:
        result["validation_note"] = fence_note.strip("; ").strip()

    return result


# ─────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "iteration", "model", "format_ok",
    "is_valid_json", "has_classification", "has_likelihood", "has_indicators",
    "validation_note", "elapsed_seconds", "input_tokens", "output_tokens",
    "raw_response", "error_message",
]


def _open_csv(model_name: str):
    os.makedirs(LLM_LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{LLM_LOGS_DIR}/output_testing_{model_name}_{timestamp}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    return path


def _append_row(path: str, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────
# EXECUÇÃO
# ─────────────────────────────────────────────────────────────────────────
def main():
    if MODEL_KEY not in LOCAL_MODELS:
        raise SystemExit(
            f"Modelo '{MODEL_KEY}' não encontrado em config.LOCAL_MODELS. "
            f"Disponíveis: {', '.join(LOCAL_MODELS)}"
        )

    if not os.path.exists(EMAIL_FILE):
        raise SystemExit(
            f"Arquivo de e-mail '{EMAIL_FILE}' não encontrado. "
            "Crie o arquivo com o e-mail a ser testado."
        )

    with open(EMAIL_FILE, "r", encoding="utf-8") as f:
        email_text = f.read()

    # Primeira linha = assunto; o restante = corpo (espelha os campos
    # 'subject' e 'body' do dataset).
    lines = email_text.splitlines()
    if lines:
        subject = lines[0].strip()
    else:
        subject = ""
    body = "\n".join(lines[1:]).strip()

    model_tag   = LOCAL_MODELS[MODEL_KEY]
    user_prompt = USER_PROMPT_TEMPLATE.format(subject=subject, body=body)

    print(f"\n{'=-'*30 + '='}")
    print(f"Teste de consistência de formato (JSON)")
    print(f"Modelo:      {MODEL_KEY} ({model_tag})")
    print(f"E-mail:      {EMAIL_FILE} (assunto: {len(subject)} chars, corpo: {len(body)} chars)")
    print(f"Repetições:  {REPETITIONS}")
    print(f"Max tokens:  {TEST_MAX_TOKENS}")
    print(f"{'=-'*30 + '='}\n")

    csv_path = _open_csv(MODEL_KEY)
    print(f"  CSV: {csv_path}\n")

    client = DmrClient(model_tag=model_tag)
    client.warm_up()

    ok_count = 0

    for i in range(1, REPETITIONS + 1):
        response = client.classify(SYSTEM_PROMPT, user_prompt,
                                    max_tokens=TEST_MAX_TOKENS)

        if response["error"]:
            v = {
                "is_valid_json":      False,
                "has_classification": False,
                "has_likelihood":     False,
                "has_indicators":     False,
                "format_ok":          False,
                "validation_note":    "Requisição falhou (ver error_message)",
            }
        else:
            v = validate_format(response["raw_response"])

        if v["format_ok"]:
            ok_count += 1
            status = "OK "
        else:
            status = "FALHA"

        print(f"  [{i:>3}/{REPETITIONS}] {status}  "
              f"({response['elapsed_seconds']}s)  {v['validation_note']}")

        _append_row(csv_path, {
            "iteration":          i,
            "model":              MODEL_KEY,
            "format_ok":          v["format_ok"],
            "is_valid_json":      v["is_valid_json"],
            "has_classification": v["has_classification"],
            "has_likelihood":     v["has_likelihood"],
            "has_indicators":        v["has_indicators"],
            "validation_note":    v["validation_note"],
            "elapsed_seconds":    response["elapsed_seconds"],
            "input_tokens":       response["input_tokens"],
            "output_tokens":      response["output_tokens"],
            "raw_response":       response["raw_response"].replace("\n", " ").replace("\r", ""),
            "error_message":      response["error"] or "",
        })

    if REPETITIONS:
        rate = ok_count / REPETITIONS * 100
    else:
        rate = 0
    print(f"\n{'=-'*30 + '='}")
    print(f"Resultado: {ok_count}/{REPETITIONS} no formato correto ({rate:.1f}%)")
    print(f"CSV salvo em: {csv_path}")
    print(f"{'=-'*30 + '='}\n")


if __name__ == "__main__":
    main()
