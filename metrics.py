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
    Recebe a lista de resultados do model_runner e retorna um dicionário
    com todas as métricas calculadas, em DUAS faixas:

    - **Bruta (estrito)**: avalia a saída como o modelo a entregou. Respostas
      que só puderam ser lidas após reparo heurístico (was_repaired) contam
      como ERRO e ficam de fora da acurácia/precisão/recall/F1. Mede a
      qualidade real da formatação do modelo (`error_rate`, `accuracy`, ...).
    - **Pós-reparo (resgatado)**: aproveita as predições resgatadas pelo reparo
      heurístico. Só sobram como erro as respostas irrecuperáveis
      (predicted_label == -1). Métricas com prefixo `rescued_`.

    A ideia é responder "houve taxa de erro X na saída bruta; ao reparar esses
    erros com heurísticas externas, os resultados passam a ser Y".

    Args:
        model_name: Nome do modelo
        results: Lista de dicionários com os resultados de cada predição
    Returns:
        Dicionário com as duas faixas de métricas (ver chaves abaixo).
    """
    total = len(results)

    # Faixa bruta: respostas limpas (sem erro algum, sem reparo).
    clean         = [r for r in results if not r["is_error"]]
    strict_errors = [r for r in results if r["is_error"]]            # inclui reparadas
    repaired_recs = [r for r in results if r.get("was_repaired")]

    # Faixa pós-reparo: qualquer predição utilizável (limpa OU resgatada).
    usable        = [r for r in results if r["predicted_label"] != -1]
    hard_errors   = [r for r in results if r["predicted_label"] == -1]  # irrecuperáveis

    error_rate         = len(strict_errors) / total if total > 0 else 0.0
    rescued_error_rate = len(hard_errors)   / total if total > 0 else 0.0
    avg_time           = sum(r["elapsed_seconds"] for r in results) / total if total > 0 else 0.0

    if not usable:
        print(f"[{model_name}] AVISO: nenhuma predição utilizável para calcular métricas.")

    strict  = _classification_metrics(clean)
    rescued = _classification_metrics(usable)

    metrics = {
        "model":         model_name,
        "total_samples": total,
        "avg_time_sec":  round(avg_time, 4),

        # ── Saída bruta (estrito): reparo conta como erro ──
        "valid_samples":   len(clean),
        "repaired_count":  len(repaired_recs),
        "error_count":     len(strict_errors),
        "error_rate":      round(error_rate, 4),
        "accuracy":        strict["accuracy"],
        "precision":       strict["precision"],
        "recall":          strict["recall"],
        "f1_score":        strict["f1_score"],
        "true_negatives":  strict["true_negatives"],
        "false_positives": strict["false_positives"],
        "false_negatives": strict["false_negatives"],
        "true_positives":  strict["true_positives"],

        # ── Pós-reparo heurístico (resgatado) ──
        "rescued_valid_samples":   len(usable),
        "rescued_error_count":     len(hard_errors),
        "rescued_error_rate":      round(rescued_error_rate, 4),
        "rescued_accuracy":        rescued["accuracy"],
        "rescued_precision":       rescued["precision"],
        "rescued_recall":          rescued["recall"],
        "rescued_f1_score":        rescued["f1_score"],
        "rescued_true_negatives":  rescued["true_negatives"],
        "rescued_false_positives": rescued["false_positives"],
        "rescued_false_negatives": rescued["false_negatives"],
        "rescued_true_positives":  rescued["true_positives"],
    }

    _print_metrics(metrics)
    _save_metrics(metrics)
    return metrics


def _classification_metrics(records: list[dict]) -> dict:
    """
    Calcula acurácia/precisão/recall/F1 + matriz de confusão para um conjunto
    de registros com predição utilizável (predicted_label em {0, 1}).
    Retorna zeros se a lista estiver vazia. Classe positiva: PHISHING (1).
    """
    if not records:
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0,
            "true_negatives": 0, "false_positives": 0,
            "false_negatives": 0, "true_positives": 0,
        }

    y_true = [r["true_label"]      for r in records]
    y_pred = [r["predicted_label"] for r in records]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision":       round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall":          round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_score":        round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
    }


def _print_metrics(m: dict):
    print(f"\n{'─'*50}")
    print(f"  Modelo:        {m['model']}")
    print(f"  Amostras:      {m['total_samples']}")
    print(f"  Tempo médio:   {m['avg_time_sec']:.2f}s / resposta")
    print(f"  ── Saída bruta (reparo conta como erro) ──")
    print(f"    Válidas:     {m['valid_samples']}/{m['total_samples']} "
          f"(reparadas: {m['repaired_count']})")
    print(f"    Taxa de erro:{m['error_rate']:.2%}")
    print(f"    Acurácia:    {m['accuracy']:.2%}   F1: {m['f1_score']:.4f}")
    print(f"    Precisão:    {m['precision']:.2%}   Recall: {m['recall']:.2%}")
    print(f"    CM:  TN={m['true_negatives']} FP={m['false_positives']} "
          f"FN={m['false_negatives']} TP={m['true_positives']}")
    print(f"  ── Pós-reparo heurístico (resgatado) ──")
    print(f"    Utilizáveis: {m['rescued_valid_samples']}/{m['total_samples']}")
    print(f"    Erro irrecuperável: {m['rescued_error_rate']:.2%}")
    print(f"    Acurácia:    {m['rescued_accuracy']:.2%}   F1: {m['rescued_f1_score']:.4f}")
    print(f"    Precisão:    {m['rescued_precision']:.2%}   Recall: {m['rescued_recall']:.2%}")
    print(f"    CM:  TN={m['rescued_true_negatives']} FP={m['rescued_false_positives']} "
          f"FN={m['rescued_false_negatives']} TP={m['rescued_true_positives']}")
    print(f"{'─'*50}")


def load_existing_metrics(model_name: str) -> dict | None:
    """
    Retorna as métricas já salvas de um modelo (reports/metrics_<modelo>.json),
    ou None se ainda não existirem. Usado para retomar benchmarks interrompidos.
    """
    path = os.path.join(REPORTS_DIR, f"metrics_{model_name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


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
    # Bruta (estrito) vs. pós-reparo, lado a lado. Só mostra colunas presentes
    # (JSONs antigos podem não ter as métricas 'rescued_*').
    cols = [c for c in [
        "model",
        "f1_score", "accuracy", "error_rate",
        "rescued_f1_score", "rescued_accuracy", "rescued_error_rate",
        "repaired_count", "avg_time_sec",
    ] if c in df.columns]
    print(df[cols].to_string())
    print(f"\nRelatório salvo em: {path}")

    return df


def rebuild_comparison() -> pd.DataFrame:
    """
    Reconstrói o relatório comparativo a partir de todos os metrics_*.json
    salvos em reports/.

    Permite rodar o benchmark em partes (rodadas separadas) e montar o
    comparativo final juntando os JSONs já existentes, sem manter tudo
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
    # Re-monta o comparativo a partir dos JSONs já existentes dos resultados individuais de cada modelo (reports/metrics_<modelo>.json).
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--rebuild":
        rebuild_comparison()
    else:
        print("Uso:\n"
              "  python metrics.py --rebuild   # remonta o comparativo")