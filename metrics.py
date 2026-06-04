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
import glob
import json
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from config import REPORTS_DIR, COMPARATIVE_REPORT_FILENAME


def compute_metrics(model_name: str, results: list[dict]) -> dict:
    """
    Recebe a lista de resultados do test_runner e retorna um dicionário
    com todas as métricas calculadas.
    Args:
        model_name: Nome do modelo
        results: Lista de dicionários com os resultados de cada predição
    Returns:
        Dicionário com métricas: accuracy, precision, recall, f1_score,
        avg_time_sec, error_rate, true_negatives, false_positives,
        false_negatives, true_positives
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
    Args:
        all_metrics: Lista de dicionários com as métricas de cada modelo
    Returns:
        DataFrame ordenado por F1-Score
    """
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    df.index += 1  # ranking começa em 1

    path = f"{COMPARATIVE_REPORT_FILENAME}"
    df.to_csv(path, index_label="ranking")

    print(f"\n{'=-'*30 + '='}")
    print("RELATÓRIO COMPARATIVO FINAL")
    print(f"\n{'=-'*30 + '='}")
    cols = ["model", "accuracy", "precision", "recall", "f1_score",
            "avg_time_sec", "error_rate"]
    print(df[cols].to_string())
    print(f"\nRelatório salvo em: {path}")

    return df


def _to_bool(v) -> bool:
    """Converte valores de CSV ('True'/'False', 1/0, etc.) para bool de verdade."""
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1")
    return bool(v)


def metrics_from_csv(csv_path: str, model_name: str | None = None) -> dict:
    """
    Recalcula as métricas a partir de um CSV salvo em results/llm_logs/.

    Útil para reprocessar uma rodada anterior ou um CSV parcial (ex.: quando
    o processo caiu no meio de um modelo) sem precisar rodar o modelo de novo.

    Args:
        csv_path: caminho do CSV gerado pelo BenchmarkLogger.
        model_name: nome do modelo. Se None, usa a coluna 'model' do CSV.
    Returns:
        Dicionário de métricas (também impresso e salvo em reports/).
    """
    df = pd.read_csv(csv_path)

    if model_name is None:
        model_name = str(df["model"].iloc[0]) if len(df) else "desconhecido"

    results = [
        {
            "true_label":      int(row["true_label"]),
            "predicted_label": int(row["predicted_label"]),
            "is_error":        _to_bool(row["is_error"]),
            "is_correct":      _to_bool(row["is_correct"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
        }
        for _, row in df.iterrows()
    ]

    print(f"Recalculando métricas de '{model_name}' a partir de {csv_path} "
          f"({len(results)} registros)")
    return compute_metrics(model_name, results)


def rebuild_comparison() -> pd.DataFrame:
    """
    Reconstrói o relatório comparativo a partir de todos os metrics_*.json
    salvos em reports/.

    Permite rodar o benchmark em partes (rodadas separadas) e montar o
    comparativo final juntando os JSONs já existentes — sem manter tudo
    em memória numa única execução.
    Returns:
        DataFrame comparativo (vazio se não houver nenhum JSON).
    """
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "metrics_*.json")))
    if not files:
        print(f"Nenhum metrics_*.json encontrado em {REPORTS_DIR}.")
        return pd.DataFrame()

    all_metrics = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            all_metrics.append(json.load(f))

    print(f"Encontrados {len(all_metrics)} relatório(s) de modelo em {REPORTS_DIR}.")
    return save_comparison_report(all_metrics)


if __name__ == "__main__":
    # Uso por linha de comando:
    #   python metrics.py <caminho_csv> [nome_modelo]   → recalcula de um CSV
    #   python metrics.py --rebuild                      → remonta o comparativo
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--rebuild":
        rebuild_comparison()
    elif len(sys.argv) >= 2:
        name = sys.argv[2] if len(sys.argv) >= 3 else None
        metrics_from_csv(sys.argv[1], name)
    else:
        print("Uso:\n"
              "  python metrics.py <caminho_csv> [nome_modelo]   # recalcula de um CSV\n"
              "  python metrics.py --rebuild                      # remonta o comparativo")