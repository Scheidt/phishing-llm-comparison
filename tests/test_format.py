#!/usr/bin/env python3
"""
Compara tres estrategias de saida estruturada para um classificador de phishing,
rodando contra o Docker Model Runner (API compativel com OpenAI):

  (a) prompt_only       -> formato pedido so no prompt, com retry/repair loop
  (b) constrained_only  -> response_format com JSON schema, prompt minimo
  (c) both              -> response_format com schema + prompt descritivo

Para cada variante mede:
  - accuracy            : classificacao bate com o rotulo (falha de formato conta como erro)
  - accuracy_parseable  : accuracy considerando so os exemplos onde saiu algo parseavel
  - format_failure_rate : fracao de exemplos sem JSON valido/completo (apos retries)
  - mean/median latency : por chamada (inclui o tempo gasto em retries)

Uso:
    pip install openai
    python tests/test_format2.py

O dataset e o endpoint vem de config.py; o modelo vem de MODEL_KEY (chave em
config.LOCAL_MODELS), igual ao test_model.py.
"""

import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field

# Permite rodar como script standalone a partir da pasta tests/:
# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI

from config import DATASET_PATH, DMR_BASE_URL, LOCAL_MODELS
from dataset.dataset import load_dataset

# --------------------------------------------------------------------------- #
# Configuracao
# --------------------------------------------------------------------------- #

MODEL_KEY = "granite4_tiny"          # chave em config.LOCAL_MODELS
MODEL = LOCAL_MODELS[MODEL_KEY]      # tag do modelo no DMR

MAX_RETRIES = 2  # tentativas extras para a variante prompt_only

# Schema com os campos na ordem "pensar antes de cravar":
# indicators -> phishing_likelihood -> classification (enum).
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "phishing_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "phishing_likelihood": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
                "classification": {
                    "type": "string",
                    "enum": ["phishing", "legitimate"],
                },
            },
            "required": ["indicators", "phishing_likelihood", "classification"],
            "additionalProperties": False,
        },
    },
}

# Prompt original (formato descrito por extenso, com bloco JSON literal).
SYSTEM_PROMPT_FULL = """You are given the full content of an email. Your task is to decide whether the
email is a phishing attempt or a legitimate message.

Analyze the email for common phishing indicators (mismatched sender/domain,
suspicious links or attachments, urgent or threatening language, requests for
sensitive information). Reason about the indicators first, then decide.

Respond ONLY with a single JSON object, with no extra text, in exactly this
format:
{
  "indicators": ["<short observation>", ...],
  "phishing_likelihood": <integer from 0 to 100>,
  "classification": "phishing" or "legitimate"
}"""

# Prompt minimo: descreve a tarefa mas NAO o formato (o grammar cuida do formato).
SYSTEM_PROMPT_MINIMAL = """You are given the full content of an email. Decide whether it is a phishing
attempt or a legitimate message. List the concrete indicators you observe,
estimate the phishing likelihood from 0 to 100, and classify it."""

USER_PROMPT_TEMPLATE = """Email to analyse:

Subject: {subject}

Body:
{body}"""


# --------------------------------------------------------------------------- #
# Resultado por chamada
# --------------------------------------------------------------------------- #

@dataclass
class CallResult:
    parsed: dict | None          # dict valido e completo, ou None se falhou
    latency: float               # segundos (soma das tentativas)
    attempts: int                # quantas chamadas a API foram feitas


def extract_json(text: str) -> dict | None:
    """Tenta extrair e validar um objeto JSON do texto livre do modelo."""
    if not text:
        return None
    # Pega do primeiro '{' ao ultimo '}', tolerando texto em volta.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    # Validacao de campos minimos.
    if "classification" not in obj or "phishing_likelihood" not in obj:
        return None
    if obj["classification"] not in ("phishing", "legitimate"):
        return None
    return obj


def normalize(obj: dict) -> dict:
    """Clampa o likelihood em 0..100 (o grammar nao garante o range de forma confiavel)."""
    try:
        likelihood = int(obj.get("phishing_likelihood", 0))
    except (TypeError, ValueError):
        likelihood = 0
    obj["phishing_likelihood"] = max(0, min(100, likelihood))
    return obj


# --------------------------------------------------------------------------- #
# As tres variantes
# --------------------------------------------------------------------------- #

def call_prompt_only(client, model, subject, body) -> CallResult:
    """Sem response_format. Pede o formato no prompt e tenta novamente se vier invalido."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FULL},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(subject=subject, body=body)},
    ]
    total_latency = 0.0
    for attempt in range(1, MAX_RETRIES + 2):  # 1 tentativa + MAX_RETRIES
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=messages,
        )
        total_latency += time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        parsed = extract_json(text)
        if parsed is not None:
            return CallResult(normalize(parsed), total_latency, attempt)
        # Reforca o pedido de formato para a proxima tentativa.
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": "That was not valid. Respond with ONLY the JSON object, nothing else.",
        })
    return CallResult(None, total_latency, MAX_RETRIES + 1)


def call_constrained(client, model, subject, body, system_prompt) -> CallResult:
    """Com response_format (grammar). system_prompt distingue (b) minimo de (c) descritivo."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(subject=subject, body=body)},
        ],
        response_format=RESPONSE_FORMAT,
    )
    latency = time.perf_counter() - t0
    parsed = extract_json(resp.choices[0].message.content or "")
    if parsed is not None:
        parsed = normalize(parsed)
    return CallResult(parsed, latency, 1)


VARIANTS = {
    "prompt_only": lambda c, m, s, b: call_prompt_only(c, m, s, b),
    "constrained_only": lambda c, m, s, b: call_constrained(c, m, s, b, SYSTEM_PROMPT_MINIMAL),
    "both": lambda c, m, s, b: call_constrained(c, m, s, b, SYSTEM_PROMPT_FULL),
}


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #

@dataclass
class Metrics:
    correct: int = 0
    parseable: int = 0
    correct_parseable: int = 0
    failures: int = 0          # chamada COMPLETOU mas JSON veio invalido/incompleto
    context_errors: int = 0    # chamada NAO completou (estouro de contexto / erro de API)
    total: int = 0
    latencies: list = field(default_factory=list)

    def completed(self):
        # exemplos em que a chamada retornou algo (exclui erros de contexto/infra)
        return self.total - self.context_errors

    def accuracy(self):
        # denominador exclui erros de contexto: eles sao ortogonais ao formato
        # e atingem todas as variantes igualmente, entao nao penalizam nenhuma.
        denom = self.completed()
        return self.correct / denom if denom else 0.0

    def accuracy_parseable(self):
        return self.correct_parseable / self.parseable if self.parseable else 0.0

    def failure_rate(self):
        # falha de formato REAL: sobre as chamadas que completaram
        denom = self.completed()
        return self.failures / denom if denom else 0.0

    def context_error_rate(self):
        return self.context_errors / self.total if self.total else 0.0


def run_variant(name, fn, client, model, dataset) -> Metrics:
    m = Metrics(total=len(dataset))
    for i, ex in enumerate(dataset, 1):
        try:
            res = fn(client, model, ex["subject"], ex["body"])
        except Exception as e:
            # Erro de contexto/API: a chamada nem completou. Balde separado,
            # porque NAO eh falha de formato e atinge as 3 variantes igual.
            kind = "contexto" if "context" in str(e).lower() or "length" in str(e).lower() else "API"
            print(f"  [{name}] exemplo {i}: erro de {kind} ({str(e)[:80]})")
            m.context_errors += 1
            continue
        m.latencies.append(res.latency)
        if res.parsed is None:
            m.failures += 1
            continue
        m.parseable += 1
        is_correct = res.parsed["classification"] == ex["label"]
        if is_correct:
            m.correct_parseable += 1
            m.correct += 1  # falha de formato ja nao chega aqui, logo conta como erro
        print(f"  [{name}] {i}/{m.total}  pred={res.parsed['classification']:<11} "
              f"gold={ex['label']:<11} {'OK' if is_correct else 'X'}  "
              f"({res.latency:.2f}s, {res.attempts} call(s))")
    return m


def print_table(results: dict):
    headers = ["variant", "accuracy", "acc_parseable", "format_fail", "context_err", "lat_mean", "lat_median"]
    widths = [18, 10, 14, 12, 12, 10, 11]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + line)
    print("-" * len(line))
    for name, m in results.items():
        lat_mean = statistics.mean(m.latencies) if m.latencies else 0.0
        lat_median = statistics.median(m.latencies) if m.latencies else 0.0
        row = [
            name,
            f"{m.accuracy():.1%}",
            f"{m.accuracy_parseable():.1%}",
            f"{m.failure_rate():.1%}",
            f"{m.context_error_rate():.1%}",
            f"{lat_mean:.2f}s",
            f"{lat_median:.2f}s",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    df = load_dataset(DATASET_PATH).head(50)  # so os 20 primeiros para teste rapido; mude ou remova para rodar tudo
    # Mapeia o rotulo inteiro do CSV (0=legitimo, 1=phishing) para a string que
    # o modelo emite, para casar com res.parsed["classification"] em run_variant.
    dataset = [
        {
            "subject": row.subject,
            "body": row.body,
            "label": "phishing" if row.label == 1 else "legitimate",
        }
        for row in df.itertuples()
    ]

    client = OpenAI(base_url=DMR_BASE_URL, api_key="not-needed")

    print(f"Modelo: {MODEL}   |   {len(dataset)} exemplos   |   endpoint: {DMR_BASE_URL}\n")

    results = {}
    for name, fn in VARIANTS.items():
        print(f"== {name} ==")
        results[name] = run_variant(name, fn, client, MODEL, dataset)

    print_table(results)

    print("Leitura:")
    print("  - accuracy: sobre as chamadas que COMPLETARAM (exclui erros de contexto).")
    print("  - acc_parseable: qualidade so quando saiu JSON valido (isola raciocinio de formato).")
    print("  - format_fail: falha de formato REAL (chamada completou mas JSON quebrou).")
    print("  - context_err: emails que estouraram o contexto / erro de API. Ortogonal ao formato,")
    print("                 atinge as 3 variantes igual -> nao penaliza nenhuma e nao entra na accuracy.")
    print("  - Compare prompt_only vs both pela accuracy; constrained_only baixo = distorcao por masking.")


if __name__ == "__main__":
    main()