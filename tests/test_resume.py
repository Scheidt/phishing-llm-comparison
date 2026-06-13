"""
Teste da retomada por e-mail (RESUME_PARTIAL_MODEL).

Objetivo
--------
Garantir que, se o benchmark de um modelo cair no meio, ao reexecutar com
`resume=True` o programa:
  1. reaproveita os e-mails já gravados no log parcial;
  2. re-testa apenas o ÚLTIMO e-mail gravado (por garantia de que não ficou
     gravado pela metade);
  3. continua gravando no MESMO arquivo de log, sem duplicar linhas;
  4. devolve, ao final, o conjunto completo de resultados (reaproveitados +
     re-testados), válido para o cálculo de métricas.

Este teste é AUTOCONTIDO: não usa o Docker Model Runner nem a API do Claude.
Ele injeta um cliente falso (FakeClient) e redireciona os logs para uma pasta
temporária, então pode rodar offline e não toca nos logs reais.

Uso
---
    python tests/test_resume.py
"""
import os
import csv
import sys
import shutil
import tempfile

# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# O console do Windows (cp1252) não imprime alguns caracteres usados nos
# relatórios (ex.: '─'); força UTF-8 para o teste rodar em qualquer terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import logger as logger_mod
import metrics as metrics_mod
import apply_thresholds
from logger import find_latest_log
from model_runner import run_model_tests
from metrics import compute_metrics

MODEL_NAME = "fake_model"
N_EMAILS   = 5


# ─────────────────────────────────────────────────────────────────────────
# CLIENTE FALSO
# ─────────────────────────────────────────────────────────────────────────
class FakeClient:
    """
    Cliente determinístico que dispensa DMR/API.

    Retorna sempre um JSON válido e registra o id de cada e-mail processado
    (extraído do user_prompt, cujo 'Subject:' é "id-<n>"). Se `crash_at` for
    informado, levanta uma exceção na enésima chamada, simulando uma queda no
    meio do benchmark.
    """
    def __init__(self, crash_at: int | None = None):
        self.crash_at = crash_at
        self.calls    = 0
        self.seen_ids = []

    def classify(self, system_prompt: str, user_prompt: str,
                 max_tokens: int | None = None) -> dict:
        self.calls += 1
        if self.crash_at is not None and self.calls == self.crash_at:
            raise RuntimeError("queda simulada no meio do benchmark")

        # extrai o id embutido no assunto ("Subject: id-<n>")
        for line in user_prompt.splitlines():
            if line.startswith("Subject: id-"):
                self.seen_ids.append(int(line.split("id-")[1].strip()))
                break

        return {
            "raw_response":    '{"classification": "phishing", '
                               '"phishing_likelihood": 90, "indicators": ["x"]}',
            "elapsed_seconds": 0.01,
            "input_tokens":    10,
            "output_tokens":   5,
            "error":           None,
        }


# ─────────────────────────────────────────────────────────────────────────
# UTILIDADES DO TESTE
# ─────────────────────────────────────────────────────────────────────────
def make_dataset() -> pd.DataFrame:
    """Dataset mínimo com ids 1..N e o id embutido no assunto."""
    return pd.DataFrame({
        "id":      list(range(1, N_EMAILS + 1)),
        "subject": [f"id-{i}" for i in range(1, N_EMAILS + 1)],
        "body":    [f"corpo do e-mail {i}" for i in range(1, N_EMAILS + 1)],
        "label":   [1, 0, 1, 0, 1],
    })


def read_log_ids(path: str) -> list[int]:
    """Lê os email_id gravados no CSV de log, na ordem."""
    with open(path, newline="", encoding="utf-8") as f:
        return [int(r["email_id"]) for r in csv.DictReader(f)]


def list_logs(logs_dir: str) -> list[str]:
    return sorted(
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir) if f.endswith(".csv")
    )


PASSED = []
FAILED = []


def check(cond: bool, msg: str):
    if cond:
        PASSED.append(msg)
        print(f"  [OK  ] {msg}")
    else:
        FAILED.append(msg)
        print(f"  [FALHA] {msg}")


# ─────────────────────────────────────────────────────────────────────────
# CENÁRIOS
# ─────────────────────────────────────────────────────────────────────────
def scenario_resume_after_crash(logs_dir: str):
    """Queda após 3 e-mails → retomada reaproveita 2 e re-testa do 3 ao 5."""
    print("\n[Cenário 1] Queda no meio + retomada")
    dataset = make_dataset()

    # 1) Roda e "cai" na 4ª chamada → log parcial com ids [1, 2, 3].
    crash_client = FakeClient(crash_at=4)
    try:
        run_model_tests(MODEL_NAME, crash_client, dataset, resume=True)
        check(False, "deveria ter levantado a queda simulada")
    except RuntimeError:
        check(True, "queda simulada interrompeu o run (como esperado)")

    logs = list_logs(logs_dir)
    check(len(logs) == 1, f"existe 1 log parcial (encontrados: {len(logs)})")
    partial_path = logs[0]
    check(read_log_ids(partial_path) == [1, 2, 3],
          f"log parcial tem ids [1,2,3] (tem: {read_log_ids(partial_path)})")

    # 2) Retoma: deve reaproveitar [1,2], re-testar [3,4,5].
    resume_client = FakeClient()
    results = run_model_tests(MODEL_NAME, resume_client, dataset, resume=True)

    check(resume_client.seen_ids == [3, 4, 5],
          f"retomada re-testou apenas ids [3,4,5] (testou: {resume_client.seen_ids})")

    logs_after = list_logs(logs_dir)
    check(len(logs_after) == 1 and logs_after[0] == partial_path,
          "retomada gravou no MESMO log (nenhum arquivo novo criado)")

    final_ids = read_log_ids(partial_path)
    check(final_ids == [1, 2, 3, 4, 5],
          f"log final tem ids [1..5] sem duplicar (tem: {final_ids})")

    check(len(results) == N_EMAILS,
          f"results cobre os {N_EMAILS} e-mails (tem: {len(results)})")

    # Pontua o log (scoring desacoplado), como faz o main.py, e então as
    # métricas precisam rodar sobre o conjunto completo sem erro de tipo.
    scored = apply_thresholds.score_log_file(partial_path)
    m = compute_metrics(MODEL_NAME, scored)
    check(m["total_samples"] == N_EMAILS and m["error_count"] == 0,
          "compute_metrics roda sobre os registros pontuados (tipos corretos)")


def scenario_resume_without_partial(logs_dir: str):
    """Sem log parcial, resume=True roda tudo do zero (sem pular nada)."""
    print("\n[Cenário 2] resume=True sem log anterior → roda tudo")
    dataset = make_dataset()
    client  = FakeClient()
    results = run_model_tests("fake_model_fresh", client, dataset, resume=True)

    check(client.seen_ids == [1, 2, 3, 4, 5],
          f"rodou os 5 e-mails do zero (testou: {client.seen_ids})")
    check(len(results) == N_EMAILS, f"results tem {N_EMAILS} registros")
    path = find_latest_log("fake_model_fresh")
    check(path is not None and read_log_ids(path) == [1, 2, 3, 4, 5],
          "log novo criado com os 5 e-mails")


# ─────────────────────────────────────────────────────────────────────────
# EXECUÇÃO
# ─────────────────────────────────────────────────────────────────────────
def main():
    tmp_dir = tempfile.mkdtemp(prefix="resume_test_")

    # Redireciona o logging para a pasta temporária e garante que está ligado.
    # (logger.py lê esses módulos-globais; basta sobrescrevê-los aqui.)
    orig_dir     = logger_mod.LLM_LOGS_DIR
    orig_enabled = logger_mod.ENABLE_LLM_LOGGING
    orig_reports = metrics_mod.REPORTS_DIR
    logger_mod.LLM_LOGS_DIR       = tmp_dir
    logger_mod.ENABLE_LLM_LOGGING = True
    metrics_mod.REPORTS_DIR       = tmp_dir  # evita poluir results/reports/

    print(f"{'=-'*30 + '='}")
    print("Teste de retomada por e-mail (RESUME_PARTIAL_MODEL)")
    print(f"Logs temporários em: {tmp_dir}")
    print(f"{'=-'*30 + '='}")

    try:
        scenario_resume_after_crash(tmp_dir)
        scenario_resume_without_partial(tmp_dir)
    finally:
        logger_mod.LLM_LOGS_DIR       = orig_dir
        logger_mod.ENABLE_LLM_LOGGING = orig_enabled
        metrics_mod.REPORTS_DIR       = orig_reports
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=-'*30 + '='}")
    print(f"Resultado: {len(PASSED)} OK, {len(FAILED)} falha(s)")
    print(f"{'=-'*30 + '='}")
    if FAILED:
        for msg in FAILED:
            print(f"  FALHOU: {msg}")
        raise SystemExit(1)
    print("Todos os testes passaram.")


if __name__ == "__main__":
    main()
