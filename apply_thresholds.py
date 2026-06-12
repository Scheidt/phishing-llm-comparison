"""
Scorer: verificação de ACERTO desacoplada da inferência.

A inferência (model_runner + logger) apenas grava os FATOS de cada resposta (a
nota `phishing_likelihood`, o classification textual, os indicators, se houve
reparo de JSON e eventual erro de chamada) e já marca a validade em is_error,
deixando vazias apenas as colunas que dependem do corte. Este módulo preenche
essas colunas:
    - predicted_label        (1 se phishing_likelihood >= corte DO MODELO, senão 0;
                              -1 se não houver nota utilizável ou houver erro)
    - predicted_label_text   ("PHISHING" / "LEGÍTIMO" / "ERRO" / "INVÁLIDO")
    - is_correct             (predicted_label == true_label)

NÃO altera is_error nem was_repaired: a validade não depende do corte e é
definida na inferência (logger.py). Dessa forma, correções manuais nos logs
(ex.: consertar um JSON reparado e marcar is_error=False) sobrevivem a um
re-scoring.

Os cortes por modelo ficam em config.PHISHING_LIKELIHOOD_THRESHOLDS (chave =
nome do modelo na coluna 'model'; modelos sem corte próprio usam "Default").
Como o rótulo deriva sempre da nota (nunca do rótulo anterior), a pontuação é
idempotente e pode ser re-executada quantas vezes for necessário.

Dois usos:
  1) No pipeline: o main.py chama score_run() logo após cada modelo, antes das
     métricas (ver score_run).
  2) Standalone: re-pontua TODOS os logs em results/llm_logs/ e recalcula as
     métricas; deve ser usado após alterar algum corte em config.py.

Uso standalone:
    python apply_thresholds.py            # re-pontua os logs e recalcula métricas
    python apply_thresholds.py --dry-run  # mostra o que mudaria, sem gravar nada
"""
import csv
import glob
import os
import sys

from config import LLM_LOGS_DIR, ENABLE_LLM_LOGGING, PHISHING_LIKELIHOOD_THRESHOLDS
from logger import BenchmarkLogger, find_latest_log, _coerce_record

DEFAULT_KEY = "Default"


def threshold_for(model_name: str) -> float:
    """Corte do modelo em config.PHISHING_LIKELIHOOD_THRESHOLDS, ou o 'Default'."""
    return PHISHING_LIKELIHOOD_THRESHOLDS.get(
        model_name, PHISHING_LIKELIHOOD_THRESHOLDS[DEFAULT_KEY]
    )


def _safe_int(value) -> int | None:
    """Converte para int, ou None em caso de falha."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_likelihood(value) -> float | None:
    """
    Devolve a nota como float 0..100, ou None se vazia/fora do intervalo.

    Aceita tanto inteiros ("92", do Claude) quanto floats ("85.0", dos modelos
    locais), pois os logs gravam a nota nos dois formatos.
    """
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if 0 <= n <= 100 else None


def score_record(rec: dict, threshold: float) -> dict:
    """
    Preenche os veredictos dependentes do corte em UM registro (in-place).

    Lê os fatos da inferência (phishing_likelihood, error_message, true_label) e
    define predicted_label, predicted_label_text e is_correct. NÃO altera
    is_error (validade, definida na inferência; ver logger.py). Funciona tanto
    para registros em memória (vindos do logger, já tipados) quanto para linhas
    lidas de um CSV (valores em string): a coerção de tipos é feita aqui.

    Regra:
      - erro de chamada OU sem nota numérica utilizável => predicted_label = -1
        (texto "ERRO" / "INVÁLIDO"), is_correct = False.
      - caso contrário => 1 se likelihood >= corte, senão 0; is_correct compara
        com true_label.
    """
    likelihood = _parse_likelihood(rec.get("phishing_likelihood", ""))
    true_label = _safe_int(rec.get("true_label"))
    has_error  = bool((rec.get("error_message") or "").strip())

    if has_error or likelihood is None:
        rec["predicted_label"]      = -1
        rec["predicted_label_text"] = "ERRO" if has_error else "INVÁLIDO"
        rec["is_correct"]           = False
    else:
        label = 1 if likelihood >= threshold else 0
        rec["predicted_label"]      = label
        rec["predicted_label_text"] = "PHISHING" if label == 1 else "LEGÍTIMO"
        rec["is_correct"]           = (label == true_label)

    return rec


def score_records(records: list[dict], threshold: float) -> list[dict]:
    """Aplica score_record a uma lista de registros (in-place) e a devolve."""
    for rec in records:
        score_record(rec, threshold)
    return records


def score_log_file(path: str, threshold: float | None = None) -> list[dict]:
    """
    Lê um CSV de log, preenche as colunas de veredicto NO LUGAR e devolve os
    registros já tipados (prontos para compute_metrics).

    Se `threshold` for None, usa o corte do modelo lido da coluna 'model'.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        return []

    if threshold is None:
        threshold = threshold_for(rows[0].get("model") or "")

    score_records(rows, threshold)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or BenchmarkLogger._csv_fields())
        writer.writeheader()
        writer.writerows(rows)

    return [_coerce_record(r) for r in rows]


def score_run(model_name: str, raw_results: list[dict]) -> list[dict]:
    """
    Pontua um modelo recém-inferido. Chamado pelo main.py logo após a inferência,
    antes das métricas.

    Com logging ligado, pontua o log no disco (fonte da verdade, gravada para
    re-scoring futuro) e devolve os registros tipados. Sem logging, pontua os
    registros em memória devolvidos pela inferência.
    """
    threshold = threshold_for(model_name)
    if ENABLE_LLM_LOGGING:
        path = find_latest_log(model_name)
        if path:
            return score_log_file(path, threshold)
    return score_records(raw_results, threshold)


def _score_and_report(path: str, dry_run: bool) -> None:
    """Pontua um log (com relatório) e o reescreve no lugar, salvo em dry-run."""
    nome = os.path.basename(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        print(f"  {nome}: vazio (pulado).")
        return

    model = rows[0].get("model") or "?"
    threshold = threshold_for(model)
    origem = "" if model in PHISHING_LIKELIHOOD_THRESHOLDS else "  (Default)"

    com_nota = mudaram = 0
    for row in rows:
        antes = str(row.get("predicted_label"))
        com_nota += _parse_likelihood(row.get("phishing_likelihood", "")) is not None
        score_record(row, threshold)
        if antes != str(row.get("predicted_label")):
            mudaram += 1

    print(f"  {model}: corte {threshold}{origem} | {len(rows)} linha(s), "
          f"{com_nota} com nota, {mudaram} alterada(s)")

    if dry_run:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or BenchmarkLogger._csv_fields())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dry_run = "--dry-run" in sys.argv[1:]

    paths = sorted(glob.glob(os.path.join(LLM_LOGS_DIR, "*.csv")))
    if not paths:
        print(f"Nenhum log .csv encontrado em {LLM_LOGS_DIR}.")
        return

    modo = "DRY-RUN (nada será gravado)" if dry_run else "gravando no lugar"
    print(f"Pontuando {len(paths)} log(s) em {LLM_LOGS_DIR} [{modo}]")
    print(f"Cortes: {PHISHING_LIKELIHOOD_THRESHOLDS}\n")

    for path in paths:
        _score_and_report(path, dry_run)

    if dry_run:
        print("\nDry-run: nenhum arquivo alterado.")
        return

    # Re-pontuou os logs no disco; agora recalcula métricas e o comparativo.
    print("\nRecalculando métricas a partir dos logs...")
    from metrics import rebuild_metrics_from_logs
    rebuild_metrics_from_logs()


if __name__ == "__main__":
    main()
