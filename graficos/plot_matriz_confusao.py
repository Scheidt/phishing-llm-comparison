"""
Gera a tabela-verdade (matriz de confusao) de cada modelo a partir do CSV de log,
aplicando sobre o phishing_likelihood o limiar otimo (menor valor que maximiza
o F1 da classe phishing), calculado por limiar.melhor_limiar_f1.

A predicao e derivada do numero: phishing se phishing_likelihood >= limiar.
Para cada modelo imprime a matriz no console e salva uma figura PNG.

Uso: ajuste a lista MODELOS abaixo (caminho do log) e rode
`python plot_matriz_confusao.py`. Para forcar um limiar manual, passe-o
explicitamente para gerar(...).
"""

# ============================================================
# (nome do modelo, caminho do log)
MODELOS = [
    ("Claude Haiku 4.5", "results/llm_logs/claude-haiku-4-5-20251001_20260607_004626.csv"),
    ("Gemma 3 4B QAT",   "results/llm_logs/gemma3_4b_qat_20260607_015900.csv"),
    ("Qwen 3 4B",        "results/llm_logs/qwen3_4b_20260607_065655.csv"),
    ("Granite 4 Tiny",   "results/llm_logs/granite4_tiny_20260607_130903.csv"),
]
# ============================================================

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from limiar import melhor_limiar_f1, carregar

# diretorios resolvidos a partir da localizacao deste script (graficos/),
# para que possa ser chamado de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
OUT_DIR = os.path.join(SCRIPT_DIR, "images")  # diretorio das figuras


def confusion(y_true, y_pred):
    """Retorna (tp, fp, tn, fn) com phishing (1) como classe positiva."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def imprimir_tabela(nome, limiar, tp, fp, tn, fn):
    """Imprime a matriz de confusao e metricas no console."""
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\n=== {nome}  (limiar = {limiar:g}) ===")
    print(f"{'':>22}| {'Pred PHISHING':>14} | {'Pred LEGITIMO':>14}")
    print(f"{'Real PHISHING':>22}| {tp:>14} | {fn:>14}")
    print(f"{'Real LEGITIMO':>22}| {fp:>14} | {tn:>14}")
    print(f"  acuracia={acc*100:.1f}%  precisao={prec*100:.1f}%  "
          f"recall={rec*100:.1f}%  F1={f1*100:.1f}%")
    return acc, prec, rec, f1


def plotar(nome, limiar, tp, fp, tn, fn, out_path):
    """Salva a matriz de confusao 2x2 como figura PNG."""
    # linhas = real (phishing, legitimo); colunas = predito (phishing, legitimo)
    matriz = np.array([[tp, fn],
                       [fp, tn]])
    rotulos = np.array([["VP", "FN"],
                        ["FP", "VN"]])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matriz, cmap="Blues")

    vmax = matriz.max() if matriz.max() else 1
    for i in range(2):
        for j in range(2):
            cor = "white" if matriz[i, j] > 0.6 * vmax else "#1A1A1A"
            ax.text(j, i, f"{rotulos[i, j]}\n{matriz[i, j]}",
                    ha="center", va="center", color=cor, fontsize=14)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["PHISHING", "LEGITIMO"])
    ax.set_yticklabels(["PHISHING", "LEGITIMO"])
    ax.set_xlabel("Predicao do modelo")
    ax.set_ylabel("Rotulo verdadeiro")
    ax.set_title(f"Matriz de confusao - {nome}\n(limiar phishing_likelihood >= {limiar:g})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  figura salva em: {out_path}")


def gerar(nome, log_path, limiar=None, out_dir=OUT_DIR):
    """Gera a matriz de confusao de um modelo (console + figura PNG).

    Se limiar for None, usa o limiar otimo (menor valor que maximiza o F1).
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(ROOT, log_path)):
        print(f"[aviso] log nao encontrado, pulando: {log_path}")
        return None

    df = pd.read_csv(os.path.join(ROOT, log_path))
    y, score, valid = carregar(df)
    y_true = y[valid].astype(int).values
    score = score[valid].astype(float).values

    if limiar is None:
        limiar, _ = melhor_limiar_f1(y_true, score)

    y_pred = (score >= limiar).astype(int)
    tp, fp, tn, fn = confusion(y_true, y_pred)
    print(f"\nLinhas validas: {int(valid.sum())} de {len(df)}", end="")
    imprimir_tabela(nome, limiar, tp, fp, tn, fn)

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", nome)
    out_path = os.path.join(out_dir, f"matriz_confusao_{slug}.png")
    plotar(nome, limiar, tp, fp, tn, fn, out_path)
    return out_path


def main():
    for nome, log_path in MODELOS:
        gerar(nome, log_path)


if __name__ == "__main__":
    main()
