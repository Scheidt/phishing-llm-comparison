"""
Sistema de logging estruturado.

Gera dois tipos de arquivo por modelo:
  1. JSONL  — um JSON por linha, com todos os detalhes de cada predição
  2. CSV    — formato tabular para análise posterior
"""
import os
import json
import csv
from datetime import datetime
from config import LOGS_DIR, REPORTS_DIR


class BenchmarkLogger:
    def __init__(self, model_name: str):
        self.model_name = model_name
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = f"{LOGS_DIR}/{model_name}_{timestamp}.jsonl"
        self.csv_path   = f"{REPORTS_DIR}/{model_name}_{timestamp}.csv"

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writeheader()

        print(f"  Logs: {self.jsonl_path}")
        print(f"  CSV:  {self.csv_path}")

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "email_id", "model", "true_label", "true_label_text",
            "raw_response", "predicted_label", "predicted_label_text",
            "is_correct", "is_error",
            "elapsed_seconds", "input_tokens", "output_tokens", "error_message",
        ]

    def log(
        self,
        email_id: int,
        true_label: int,
        raw_response: str,
        predicted: str | None,
        elapsed: float,
        input_tokens: int,
        output_tokens: int,
        error: str | None,
    ) -> dict:
        """Registra o resultado de uma predição individual."""
        true_text = "PHISHING" if true_label == 1 else "LEGÍTIMO"

        if error or predicted is None:
            pred_label = -1
            pred_text  = "ERRO" if error else "INVÁLIDO"
            is_correct = False
            is_error   = True
        else:
            pred_label = 1 if predicted == "PHISHING" else 0
            pred_text  = predicted
            is_correct = (pred_label == true_label)
            is_error   = False

        record = {
            "email_id":             email_id,
            "model":                self.model_name,
            "true_label":           true_label,
            "true_label_text":      true_text,
            "raw_response":         raw_response,
            "predicted_label":      pred_label,
            "predicted_label_text": pred_text,
            "is_correct":           is_correct,
            "is_error":             is_error,
            "elapsed_seconds":      elapsed,
            "input_tokens":         input_tokens,
            "output_tokens":        output_tokens,
            "error_message":        error or "",
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writerow(record)

        return record