"""
Sistema de logging.

Gera um log dos resultados das predições e salva no local especificado.
"""
import os
import csv
import glob
from datetime import datetime
from config import LLM_LOGS_DIR, ENABLE_LLM_LOGGING


class BenchmarkLogger:
    def __init__(self, model_name: str, resume_path: str | None = None):
        """
        Args:
            model_name: nome do modelo.
            resume_path: se informado, continua gravando neste CSV existente
                         (retomada), sem recriar o arquivo nem o cabeçalho.
        """
        self.model_name = model_name
        self.csv_path = None

        if not ENABLE_LLM_LOGGING:
            return

        os.makedirs(LLM_LOGS_DIR, exist_ok=True)

        if resume_path is not None:
            # Retomada: continua no mesmo log (o cabeçalho já existe).
            self.csv_path = resume_path
            print(f"  CSV (retomando):  {self.csv_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = f"{LLM_LOGS_DIR}/{model_name}_{timestamp}.csv"

            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields())
                writer.writeheader()

            print(f"  CSV:  {self.csv_path}")

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "email_id", "email_from", "model", "true_label", "true_label_text",
            "raw_response", "predicted_label", "predicted_label_text",
            "predicted_classification_text",
            "phishing_likelihood", "indicators", "parse_note",
            "is_correct", "is_error", "was_repaired",
            "elapsed_seconds", "input_tokens", "output_tokens", "error_message",
        ]

    def log(
        self,
        email_id: int,
        email_from: str,
        true_label: int,
        raw_response: str,
        phishing_likelihood: int | None,
        indicators: list[str],
        parse_note: str,
        elapsed: float,
        input_tokens: int,
        output_tokens: int,
        error: str | None,
        repaired: bool = False,
        classification_text: str | None = None,
    ) -> dict:
        """
        Registra os FATOS de uma resposta individual, SEM julgar acerto.

        A inferência coleta os fatos da resposta (a nota numérica
        phishing_likelihood, o classification textual, os indicators, se houve
        reparo de JSON em was_repaired e eventual erro de chamada em
        error_message) e marca a VALIDADE da resposta em is_error (erro de
        chamada, OU JSON reparado, OU sem nota numérica utilizável). A validade
        não depende do corte, então é definida aqui; por ser uma coluna
        estável, pode ser corrigida à mão (ex.: consertar um JSON reparado e
        marcar is_error=False) sem que o scorer a sobrescreva.

        As colunas de VEREDICTO que dependem do corte (predicted_label,
        predicted_label_text, is_correct) ficam VAZIAS aqui e são preenchidas
        depois pelo scorer (apply_thresholds.py). Esse desenho desacopla a
        coleta do julgamento e permite re-decidir o rótulo alterando apenas os
        cortes.

        Sobre is_error vs was_repaired: was_repaired é o fato cru ("o JSON
        precisou de reparo?"); is_error é a validade agregada. O reparo conta
        como erro na faixa estrita das métricas, mas a predição resgatada ainda
        é aproveitada na faixa pós-reparo (ver metrics.py).
        """
        if true_label == 1:
            true_text = "PHISHING"
        else:
            true_text = "LEGÍTIMO"

        # Validade da resposta (não depende do corte): erro de chamada, JSON
        # reparado, ou ausência de nota numérica utilizável.
        is_error = bool(error) or repaired or (phishing_likelihood is None)

        # No CSV, a ausência de nota é gravada como célula vazia.
        if phishing_likelihood is None:
            likelihood_csv = ""
        else:
            likelihood_csv = phishing_likelihood

        record = {
            "email_id":             email_id,
            "email_from":           (email_from or "").replace("\n", " ").replace("\r", ""),
            "model":                self.model_name,
            "true_label":           true_label,
            "true_label_text":      true_text,
            "raw_response":         raw_response.replace("\n", " ").replace("\r", ""),
            # Veredictos (dependem do corte) preenchidos depois pelo scorer:
            "predicted_label":      "",
            "predicted_label_text": "",
            # classification textual do próprio modelo (não decide o rótulo;
            # registrado para análise).
            "predicted_classification_text": classification_text or "",
            "phishing_likelihood":  likelihood_csv,
            "indicators":           " | ".join(indicators),
            "parse_note":           parse_note,
            "is_correct":           "",
            "is_error":             is_error,
            "was_repaired":         repaired,
            "elapsed_seconds":      elapsed,
            "input_tokens":         input_tokens,
            "output_tokens":        output_tokens,
            "error_message":        error or "",
        }

        if self.csv_path is not None:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fields())
                writer.writerow(record)

        return record


def find_latest_log(model_name: str) -> str | None:
    """Retorna o CSV de log mais recente deste modelo em LLM_LOGS_DIR, ou None."""
    matches = sorted(glob.glob(os.path.join(LLM_LOGS_DIR, f"{model_name}_*.csv")))
    if not matches:
        return None
    return matches[-1]


def _coerce_record(row: dict) -> dict:
    """Converte os tipos de uma linha lida do CSV de volta para o formato dos records."""
    def to_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    rec = dict(row)
    rec["true_label"]      = to_int(row.get("true_label"))
    rec["predicted_label"] = to_int(row.get("predicted_label"), -1)
    rec["elapsed_seconds"] = to_float(row.get("elapsed_seconds"))
    rec["input_tokens"]    = to_int(row.get("input_tokens"))
    rec["output_tokens"]   = to_int(row.get("output_tokens"))
    rec["is_correct"]      = str(row.get("is_correct")).strip().lower() == "true"
    rec["is_error"]        = str(row.get("is_error")).strip().lower() == "true"
    rec["was_repaired"]    = str(row.get("was_repaired")).strip().lower() == "true"
    return rec


def load_partial_results(model_name: str) -> tuple[str | None, list[dict], set[int]]:
    """
    Prepara a retomada de um modelo interrompido a partir do seu log parcial.

    Reaproveita todos os e-mails já gravados, EXCETO o último (que será
    re-testado, para garantir que foi gravado corretamente). Reescreve o CSV
    sem essa última linha, de modo que a retomada continue gravando no mesmo
    arquivo, sem duplicar o e-mail re-testado.

    Returns:
        (csv_path, records, skip_ids)
        csv_path: log a reutilizar (None se não houver log parcial / logging off)
        records:  registros já válidos a reaproveitar no cálculo de métricas
        skip_ids: ids de e-mail a pular na iteração do dataset
    """
    if not ENABLE_LLM_LOGGING:
        return None, [], set()

    path = find_latest_log(model_name)
    if path is None:
        return None, [], set()

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return path, [], set()

    # Descarta a última linha (último e-mail testado) para re-testá-la.
    keep = rows[:-1]
    skip_ids = {int(r["email_id"]) for r in keep}

    # Reescreve o CSV sem a última linha, para a retomada continuar no
    # mesmo arquivo sem duplicar o e-mail que será re-testado.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BenchmarkLogger._csv_fields())
        writer.writeheader()
        writer.writerows(keep)

    records = [_coerce_record(r) for r in keep]
    print(f"  Retomando '{model_name}': {len(records)} e-mail(s) reaproveitado(s), "
          f"re-testando a partir do id {rows[-1]['email_id']}.")
    return path, records, skip_ids