"""
Gera UM grafico com a curva ROC de TODOS os modelos sobreposta, cada modelo
com uma cor diferente. O AUC de cada curva aparece na legenda, e as curvas sao
ordenadas por AUC (maior primeiro) para facilitar a leitura.

O catalogo de modelos (apelido -> nome de exibicao, caminho do log) e lido de
plot.py (MODELOS_DISPONIVEIS / MODELOS); a selecao dos modelos exibidos e
editada la. Execute `python plot_roc_todos.py` a partir de qualquer diretorio.
A figura e salva em graficos/images/roc_todos.png.
"""

# ============================================================
OUT_PATH = None        # None = graficos/images/roc_todos.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from limiar import carregar
from plot_roc import roc_curve_points, auc_trapz
from plot import MODELOS_DISPONIVEIS, MODELOS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

# paleta de cores qualitativa (uma cor por modelo); cicla se houver mais modelos
CORES = [
    "#185FA5",  # azul
    "#D85A30",  # laranja
    "#0F6E56",  # verde
    "#992D2D",  # vermelho
    "#6A4C93",  # roxo
    "#C49A00",  # mostarda
    "#1B998B",  # turquesa
    "#A6324E",  # vinho
]


def _curva_modelo(log_path):
    """Carrega o log e devolve (roc, auc) usando o mesmo filtro dos outros graficos."""
    df = pd.read_csv(os.path.join(ROOT, log_path))
    y_all, score_all, valid = carregar(df)
    y = y_all[valid].astype(int).values
    score = score_all[valid].astype(float).values
    roc = roc_curve_points(y, score)
    return roc, auc_trapz(roc), int(valid.sum()), len(df)


def gerar(modelos=MODELOS, out_path=None, show=False):
    # computa cada curva primeiro, para poder ordenar por AUC na legenda
    curvas = []
    for apelido in modelos:
        if apelido not in MODELOS_DISPONIVEIS:
            print(f"[aviso] modelo desconhecido, pulando: {apelido}")
            continue
        nome, log_path = MODELOS_DISPONIVEIS[apelido]
        try:
            roc, auc, n_validas, n_total = _curva_modelo(log_path)
        except FileNotFoundError:
            print(f"[aviso] log nao encontrado, pulando: {log_path}")
            continue
        print(f"{nome}: AUC={auc:.3f}  ({n_validas}/{n_total} linhas validas)")
        curvas.append((nome, roc, auc))

    if not curvas:
        print("[erro] nenhum modelo valido para plotar.")
        return None

    curvas.sort(key=lambda c: c[2], reverse=True)   # maior AUC primeiro

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for i, (nome, roc, auc) in enumerate(curvas):
        cor = CORES[i % len(CORES)]
        ax.plot(roc[:, 0], roc[:, 1], color=cor, lw=2,
                label=f"{nome}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#B4B2A9", ls="--", lw=1, label="Aleatorio")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Taxa de falsos positivos  (1 - especificidade)")
    ax.set_ylabel("Recall de phishing  (TPR)")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "roc_todos.png")
    fig.savefig(out, dpi=150)
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    gerar(MODELOS, OUT_PATH, show=True)


if __name__ == "__main__":
    main()
