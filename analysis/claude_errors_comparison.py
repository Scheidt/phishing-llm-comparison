#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise 01 — Desempenho dos demais modelos nos e-mails que o Claude ERROU.

Pergunta respondida:
    Pegando o subconjunto de e-mails em que o Claude errou, como os outros
    modelos (gemma3, granite4, qwen3) se saíram NESSES MESMOS e-mails?
    Isso indica se os erros do Claude são "difíceis" (todos erram) ou
    idiossincráticos (os outros acertam).

Convenções (alinhadas ao formato dos logs):
    - Classe positiva = phishing (1); legítimo = 0.
    - predicted_label == -1  => ERRO DE FORMATO (o modelo não devolveu uma
      resposta parseável). NÃO entra no cálculo de TP/TN/FP/FN; é reportado
      separadamente como "taxa de inválidos".
    - Erro de CLASSIFICAÇÃO = predição válida (0/1) porém diferente do rótulo
      verdadeiro. É o "modelo falhou em identificar".
    - A decisão (likelihood >= 50 -> phishing) já está embutida em
      predicted_label, então usamos predicted_label diretamente.

Os e-mails são casados entre modelos pela coluna email_id.
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# CONSTANTES — ajuste os caminhos dos quatro logs aqui
# ============================================================
CLAUDE_LOG   = "logs/claude.csv"
GEMMA3_LOG   = "logs/gemma3.csv"
GRANITE4_LOG = "logs/granite4.csv"
QWEN3_LOG    = "logs/qwen3.csv"

# Ordem/rótulos usados nos relatórios
MODEL_LOGS = {
    "Claude":   CLAUDE_LOG,
    "gemma3":   GEMMA3_LOG,
    "granite4": GRANITE4_LOG,
    "qwen3":    QWEN3_LOG,
}

# Modelo de referência cujos erros definem o subconjunto analisado
REFERENCE_MODEL = "Claude"

# Como definir "erro do Claude" que monta o subconjunto:
#   "classification" -> apenas erros de classificação (predição válida 0/1 porém errada)
#   "all"            -> inclui também os erros de formato (-1) do Claude
CLAUDE_ERROR_MODE = "classification"

POSITIVE_LABEL = 1
INVALID_LABEL  = -1

# Onde salvar o resumo em CSV (None para não salvar)
SUMMARY_CSV_OUT = "resumo_analise_01.csv"


# ============================================================
# Funções utilitárias
# ============================================================
def safe_div(n, d):
    """Divisão que devolve NaN em vez de estourar quando o denominador é 0."""
    return (n / d) if d else float("nan")


def load_log(path):
    """Carrega um log .csv garantindo os tipos das colunas usadas."""
    df = pd.read_csv(path)
    needed = {"email_id", "true_label", "predicted_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltam colunas {missing}")
    df["email_id"] = df["email_id"].astype(int)
    df["true_label"] = df["true_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)
    return df


def get_reference_errors(df_ref, mode):
    """Retorna o DataFrame de e-mails em que o modelo de referência errou."""
    valid = df_ref["predicted_label"] != INVALID_LABEL
    wrong = df_ref["predicted_label"] != df_ref["true_label"]
    if mode == "classification":
        mask = valid & wrong
    elif mode == "all":
        mask = wrong  # -1 também é != true_label, então entra
    else:
        raise ValueError("CLAUDE_ERROR_MODE deve ser 'classification' ou 'all'")
    return df_ref[mask]


def compute_metrics(df, email_ids):
    """Métricas de classificação de `df` restritas ao conjunto de email_ids."""
    sub = df[df["email_id"].isin(email_ids)]
    total = len(sub)

    invalid = sub["predicted_label"] == INVALID_LABEL
    n_invalid = int(invalid.sum())
    valid = sub[~invalid]
    n_valid = len(valid)

    yt = valid["true_label"]
    yp = valid["predicted_label"]
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    return {
        "n_alvo":       total,        # e-mails do subconjunto presentes neste log
        "n_validos":    n_valid,
        "n_invalidos":  n_invalid,    # erros de FORMATO (separados)
        "taxa_inval":   safe_div(n_invalid, total),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "n_corretos":   tp + tn,
        "accuracy":     safe_div(tp + tn, n_valid),
        "precision":    safe_div(tp, tp + fp),
        "recall":       safe_div(tp, tp + fn),
        "specificity":  safe_div(tn, tn + fp),
        "f1":           safe_div(2 * tp, 2 * tp + fp + fn),
    }


def fmt_pct(x):
    return "  n/a " if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x*100:5.1f}%"


# ============================================================
# Programa principal
# ============================================================
def main():
    # Carrega o modelo de referência (obrigatório)
    ref_path = MODEL_LOGS[REFERENCE_MODEL]
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Log do modelo de referência ({REFERENCE_MODEL}) não encontrado: {ref_path}"
        )
    df_ref = load_log(ref_path)

    # Define o subconjunto de erros do Claude
    err = get_reference_errors(df_ref, CLAUDE_ERROR_MODE)
    error_ids = set(err["email_id"])

    # Caracterização dos erros do Claude (formato x classificação, FP x FN)
    n_format = int((df_ref["predicted_label"] == INVALID_LABEL).sum())
    valid_ref = df_ref[df_ref["predicted_label"] != INVALID_LABEL]
    n_class_err = int((valid_ref["predicted_label"] != valid_ref["true_label"]).sum())
    fp_ref = int(((valid_ref["true_label"] == 0) & (valid_ref["predicted_label"] == 1)).sum())
    fn_ref = int(((valid_ref["true_label"] == 1) & (valid_ref["predicted_label"] == 0)).sum())

    print("=" * 78)
    print(f"ANÁLISE 01 — Desempenho dos modelos nos e-mails que o {REFERENCE_MODEL} errou")
    print("=" * 78)
    print(f"Total de e-mails no log do {REFERENCE_MODEL}: {len(df_ref)}")
    print(f"Modo de seleção do subconjunto: CLAUDE_ERROR_MODE = '{CLAUDE_ERROR_MODE}'")
    print()
    print(f"Caracterização dos erros do {REFERENCE_MODEL}:")
    print(f"  - Erros de FORMATO (predicted = -1) ............ {n_format}")
    print(f"  - Erros de CLASSIFICAÇÃO (válido porém errado) . {n_class_err}")
    print(f"        * Falsos POSITIVOS (legítimo -> phishing) . {fp_ref}")
    print(f"        * Falsos NEGATIVOS (phishing -> legítimo) . {fn_ref}")
    print()
    print(f">>> Subconjunto analisado: {len(error_ids)} e-mails "
          f"({'só classificação' if CLAUDE_ERROR_MODE=='classification' else 'classificação + formato'})")
    if len(error_ids) == 0:
        print("Nenhum erro no subconjunto — nada a comparar.")
        return

    # Composição de classes do subconjunto (mesma para todos os modelos)
    sub_ref = df_ref[df_ref["email_id"].isin(error_ids)]
    n_pos = int((sub_ref["true_label"] == 1).sum())
    n_neg = int((sub_ref["true_label"] == 0).sum())
    print(f"    Composição real: {n_pos} phishing (1) / {n_neg} legítimos (0)")
    print()

    # Calcula métricas dos demais modelos nesse mesmo subconjunto
    other_models = [m for m in MODEL_LOGS if m != REFERENCE_MODEL]
    rows = []
    for name in other_models:
        path = MODEL_LOGS[name]
        if not os.path.exists(path):
            print(f"[aviso] log de {name} não encontrado ({path}); pulando.")
            continue
        df = load_log(path)
        m = compute_metrics(df, error_ids)
        m["modelo"] = name
        rows.append(m)

    if not rows:
        print("Nenhum outro modelo disponível para comparação.")
        return

    # Tabela 1: acerto bruto no subconjunto
    print("-" * 78)
    print(f"Quantos dos {len(error_ids)} e-mails (que o {REFERENCE_MODEL} errou) cada modelo ACERTOU:")
    print("-" * 78)
    print(f"{'modelo':<10}{'acertos':>9}{'% acerto':>10}{'inválidos':>11}{'% inváli':>10}")
    for m in rows:
        pct_ok = safe_div(m["n_corretos"], m["n_validos"])
        print(f"{m['modelo']:<10}{m['n_corretos']:>9}{fmt_pct(pct_ok):>10}"
              f"{m['n_invalidos']:>11}{fmt_pct(m['taxa_inval']):>10}")
    print()

    # Tabela 2: métricas de classificação completas no subconjunto
    print("-" * 78)
    print("Métricas de classificação no subconjunto (só sobre predições válidas):")
    print("-" * 78)
    print(f"{'modelo':<10}{'acc':>8}{'prec':>8}{'recall':>8}{'spec':>8}{'F1':>8}"
          f"{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}")
    for m in rows:
        print(f"{m['modelo']:<10}{fmt_pct(m['accuracy']):>8}{fmt_pct(m['precision']):>8}"
              f"{fmt_pct(m['recall']):>8}{fmt_pct(m['specificity']):>8}{fmt_pct(m['f1']):>8}"
              f"{m['TP']:>5}{m['TN']:>5}{m['FP']:>5}{m['FN']:>5}")
    print()
    print("Leitura: 'recall' alto = pega os phishing que o Claude deixou passar;")
    print("         'spec' alto   = não confunde os legítimos que o Claude marcou errado.")

    # Salva resumo
    if SUMMARY_CSV_OUT:
        cols = ["modelo", "n_alvo", "n_validos", "n_invalidos", "taxa_inval",
                "n_corretos", "accuracy", "precision", "recall", "specificity",
                "f1", "TP", "TN", "FP", "FN"]
        out = pd.DataFrame(rows)[cols]
        out.to_csv(SUMMARY_CSV_OUT, index=False)
        print(f"\nResumo salvo em: {SUMMARY_CSV_OUT}")


if __name__ == "__main__":
    main()