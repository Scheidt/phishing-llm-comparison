"""
Cálculo de métricas de desempenho para cada modelo.

Métricas calculadas:
  - Acurácia
  - Precisão  (classe positiva: PHISHING = 1)
  - Recall    (classe positiva: PHISHING = 1)
  - F1-Score
  - Tempo médio de resposta (segundos)
  - Taxa de erro (% de respostas inválidas ou com falha)
"""
import json
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from config import REPORTS_DIR


def compute_metrics(model_name: str, results: list[dict]) -> dict:
    """
    Recebe a lista de resultados do test_runner e retorna um dicionário
    com todas as métricas calculadas.
    """
    total  = len(results)
    errors = [r for r in results if r["is_error"]]
    valid  = [r for r in results if not r["is_error"]]

    error_rate = len(errors) / total if total > 0 else 0.0
    avg_time   = sum(r["elapsed_seconds"] for r in results) / total if total > 0 else 0.0

    if not valid:
        print(f"[{model_name}] AVISO: nenhuma predição válida para calcular métricas.")
        return _empty_metrics(model_name, total, error_rate, avg_time)

    y_true = [r["true_label"]      for r in valid]
    y_pred = [r["predicted_label"] for r in valid]

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(   y_true, y_pred, pos_label=1, zero_division=0)
    f1        = f1_score(       y_true, y_pred, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "model":           model_name,
        "total_samples":   total,
        "valid_samples":   len(valid),
        "error_count":     len(errors),
        "error_rate":      round(error_rate, 4),
        "accuracy":        round(accuracy,   4),
        "precision":       round(precision,  4),
        "recall":          round(recall,     4),
        "f1_score":        round(f1,         4),
        "avg_time_sec":    round(avg_time,   4),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
    }

    _print_metrics(metrics)
    _save_metrics(metrics)
    return metrics


def _empty_metrics(model_name, total, error_rate, avg_time) -> dict:
    return {
        "model": model_name, "total_samples": total, "valid_samples": 0,
        "error_count": total, "error_rate": error_rate,
        "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0,
        "avg_time_sec": avg_time,
        "true_negatives": 0, "false_positives": 0,
        "false_negatives": 0, "true_positives": 0,
    }


def _print_metrics(m: dict):
    print(f"\n{'─'*50}")
    print(f"  Modelo:        {m['model']}")
    print(f"  Amostras:      {m['valid_samples']}/{m['total_samples']} válidas")
    print(f"  Acurácia:      {m['accuracy']:.2%}")
    print(f"  Precisão:      {m['precision']:.2%}")
    print(f"  Recall:        {m['recall']:.2%}")
    print(f"  F1-Score:      {m['f1_score']:.4f}")
    print(f"  Tempo médio:   {m['avg_time_sec']:.2f}s / resposta")
    print(f"  Taxa de erro:  {m['error_rate']:.2%}")
    print(f"  Matriz de Confusão:")
    print(f"    TN={m['true_negatives']}  FP={m['false_positives']}")
    print(f"    FN={m['false_negatives']}  TP={m['true_positives']}")
    print(f"{'─'*50}")


def _save_metrics(m: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = f"{REPORTS_DIR}/metrics_{m['model']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"  Métricas salvas em: {path}")


def save_comparison_report(all_metrics: list[dict]) -> pd.DataFrame:
    """
    Gera um CSV comparativo com todos os modelos lado a lado.
    Ideal para incluir diretamente no TCC.
    """
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    df.index += 1  # ranking começa em 1

    path = f"{REPORTS_DIR}/comparison_report.csv"
    df.to_csv(path, index_label="ranking")

    print(f"\n{'='*60}")
    print("RELATÓRIO COMPARATIVO FINAL")
    print(f"{'='*60}")
    cols = ["model", "accuracy", "precision", "recall", "f1_score",
            "avg_time_sec", "error_rate"]
    print(df[cols].to_string())
    print(f"\nRelatório salvo em: {path}")

    return df