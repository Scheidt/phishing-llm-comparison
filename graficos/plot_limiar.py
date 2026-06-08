"""
Plota o F1 da classe phishing em funcao do limiar aplicado ao phishing_likelihood,
marcando com um ponto o limiar que maximiza o F1.

Uso: ajuste LOG_PATH abaixo e rode `python plot_f1_threshold.py`.
"""

# ============================================================
LOG_PATH = "results/llm_logs/qwen3_4b_20260607_065655.csv"   # <-- caminho do log
MODEL_NAME = "Qwen 3 4B"
OUT_PATH = None        # None = deriva do nome do modelo (ex.: f1_threshold_<modelo>.png)
STEP = 0.5             # resolucao da varredura do limiar (0..100)
# ============================================================

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from limiar import f1_phishing, melhor_limiar_f1, carregar

# diretorios resolvidos a partir da localizacao deste script (graficos/),
# para que possa ser chamado de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")


def gerar(model_name=MODEL_NAME, log_path=LOG_PATH, out_path=None, step=STEP, show=False):
    df = pd.read_csv(os.path.join(ROOT, log_path))

    y, score, valid = carregar(df)
    y = y[valid].astype(int).values
    score = score[valid].astype(float).values
    print(f"Linhas validas: {valid.sum()} de {len(df)}")

    # curva F1 x limiar: grade fina apenas para desenhar a linha suave
    thresholds = np.arange(0.0, 100.0 + step, step)
    f1s = np.array([f1_phishing(y, (score >= t).astype(int)) for t in thresholds])

    # ponto otimo: menor limiar que maximiza o F1 (ponto medio do plato otimo)
    best_t, best_f1 = melhor_limiar_f1(y, score)
    print(f"Melhor F1 = {best_f1*100:.2f}% no limiar {best_t:g}")

    model = model_name

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s * 100, color="#185FA5", lw=2)
    ax.axvline(best_t, color="#B4B2A9", ls="--", lw=1)
    ax.scatter([best_t], [best_f1 * 100], s=130, color="#0F6E56",
               edgecolors="white", linewidths=1.2, zorder=5,
               label=f"melhor F1 = {best_f1*100:.1f}%  (limiar = {best_t:g})")
    ax.annotate(f"{best_f1*100:.1f}%\n@ {best_t:g}",
                xy=(best_t, best_f1 * 100),
                xytext=(8, -28), textcoords="offset points",
                fontsize=10, color="#0F6E56")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Limiar sobre phishing_likelihood")
    ax.set_ylabel("F1 da classe phishing (%)")
    ax.set_title(f"F1 x limiar - {model}")
    ax.legend(loc="lower center", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(
        IMAGES_DIR, f"f1_threshold_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.png")
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