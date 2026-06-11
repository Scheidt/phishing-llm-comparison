"""
Plota o phishing_likelihood (eixo Y, 0-100) de cada e-mail contra o seu
email_id (eixo X), colorindo os pontos pelo rotulo verdadeiro (legitimo x
phishing). Tracam-se tambem o limiar que maximiza o F1 (linha tracejada) e
o corte fixo em 50, para visualizar quais e-mails caem de cada lado.

Uso: ajuste LOG_PATH abaixo e rode `python plot_likelihood_por_id.py`.
"""

# ============================================================
LOG_PATH = "results/llm_logs/qwen3_4b_20260607_065655.csv"   # <-- caminho do log
MODEL_NAME = "Qwen 3 4B"
OUT_PATH = None        # None = deriva do nome do modelo (ex.: likelihood_por_id_<modelo>.png)
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
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

COL_ID = "email_id"   # identificador do e-mail (eixo X)


def gerar(model_name=MODEL_NAME, log_path=LOG_PATH, out_path=None, show=False):
    df = pd.read_csv(os.path.join(ROOT, log_path))

    y_all, score_all, valid = carregar(df)
    # eixo X: usa o email_id se numerico; senao, o indice da linha
    ids_all = pd.to_numeric(df[COL_ID], errors="coerce") if COL_ID in df.columns else pd.Series(np.nan, index=df.index)
    if ids_all.isna().all():
        ids_all = pd.Series(np.arange(len(df)), index=df.index)

    y = y_all[valid].astype(int).values
    score = score_all[valid].astype(float).values
    ids = ids_all[valid].astype(float).values
    print(f"Linhas validas: {valid.sum()} de {len(df)}")

    # limiar otimo (menor que maximiza o F1 de phishing)
    best_t, best_f1 = melhor_limiar_f1(y, score)
    print(f"Melhor F1 = {best_f1*100:.2f}% no limiar {best_t:g}")

    model = model_name

    # plot: um ponto por e-mail, cor pelo rotulo verdadeiro
    fig, ax = plt.subplots(figsize=(11, 5))
    leg = y == 0
    phi = y == 1
    ax.scatter(ids[leg], score[leg], s=28, color="#185FA5", alpha=0.7,
               edgecolors="white", linewidths=0.4, label=f"Legitimo (n={leg.sum()})")
    ax.scatter(ids[phi], score[phi], s=28, color="#D85A30", alpha=0.7,
               edgecolors="white", linewidths=0.4, label=f"Phishing (n={phi.sum()})")

    ax.axhline(best_t, color="#00A86B", ls="--", lw=2.2,
               label=f"limiar F1-otimo = {best_t:g}")
    ax.axhline(50, color="#8E44AD", ls=":", lw=2.0, label="limiar 50")

    ax.set_ylim(-2, 102)
    ax.set_xlabel("email_id")
    ax.set_ylabel("phishing_likelihood (0-100)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(
        IMAGES_DIR, f"likelihood_por_id_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.png")
    fig.savefig(out, dpi=150)
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    gerar(MODEL_NAME, LOG_PATH, OUT_PATH, show=True)


if __name__ == "__main__":
    main()
