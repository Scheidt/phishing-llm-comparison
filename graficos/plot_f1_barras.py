"""
Grafico de barras comparando o F1 da classe phishing de cada modelo nas tres
regras operacionais usadas na curva ROC:

  1. classification  -> rotulo textual do modelo (coluna predicted_classification_text)
  2. likelihood >= 50 -> corte fixo no meio da escala
  3. F1-otimo         -> corte que maximiza o F1 (menor limiar do plato otimo)

As barras ficam agrupadas por modelo (as 3 regras lado a lado para cada modelo).
Uso: ajuste a selecao de modelos em plot.py (MODELOS / MODELOS_DISPONIVEIS) e
rode `python plot_f1_barras.py`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from limiar import melhor_limiar_f1, carregar
from plot_roc import text_to_label, classification_from_raw, rates, COL_CLS, COL_RAW

# diretorios resolvidos a partir da localizacao deste script (graficos/),
# para que possa ser chamado de qualquer diretorio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

OUT_PATH = None        # None = salva em images/f1_barras_comparacao.png

# rotulos, cores e marcadores das tres regras (mesmas cores da curva ROC)
REGRAS = [
    ("classification", "#D85A30"),
    ("likelihood >= 50", "#992D2D"),
    ("F1-otimo", "#0F6E56"),
]


def f1s_do_modelo(log_path):
    """Retorna (f1_classification, f1_limiar50, f1_otimo, t_otimo) de um log."""
    df = pd.read_csv(os.path.join(ROOT, log_path))

    # rotulo textual do modelo (Regra 1), com fallback para o raw_response
    if COL_CLS in df.columns:
        cls = df[COL_CLS].apply(text_to_label)
    else:
        print(f"[aviso] coluna '{COL_CLS}' ausente; extraindo de '{COL_RAW}'.")
        cls = df[COL_RAW].apply(classification_from_raw)

    # mesmo filtro base dos demais graficos -> mesmo limiar otimo
    y_all, score_all, valid = carregar(df)
    y = y_all[valid].astype(int).values
    score = score_all[valid].astype(float).values

    # Regra 1 exige tambem o rotulo textual parseavel
    valid_cls = valid & cls.notna()
    y_cls = y_all[valid_cls].astype(int).values
    cls_v = cls[valid_cls].astype(int).values

    t_opt, _ = melhor_limiar_f1(y, score)
    f1_cls = rates(y_cls, cls_v)[2]
    f1_50 = rates(y, (score >= 50).astype(int))[2]
    f1_opt = rates(y, (score >= t_opt).astype(int))[2]
    return f1_cls, f1_50, f1_opt, t_opt


def gerar(modelos, out_path=None, show=False):
    """modelos: lista de (nome_exibicao, log_path)."""
    nomes = []
    f1_matriz = []   # uma linha por modelo: [f1_cls, f1_50, f1_opt]
    for nome, log_path in modelos:
        f1_cls, f1_50, f1_opt, t_opt = f1s_do_modelo(log_path)
        nomes.append(nome)
        f1_matriz.append([f1_cls * 100, f1_50 * 100, f1_opt * 100])
        print(f"{nome}: classification={f1_cls*100:.1f}  "
              f"likelihood>=50={f1_50*100:.1f}  F1-otimo(t={t_opt:g})={f1_opt*100:.1f}")

    f1_matriz = np.array(f1_matriz)          # shape (n_modelos, 3)
    n_modelos = len(nomes)
    n_regras = len(REGRAS)

    x = np.arange(n_modelos)                 # posicao de cada grupo (modelo)
    largura = 0.8 / n_regras                 # largura de cada barra dentro do grupo

    fig, ax = plt.subplots(figsize=(2.6 * n_modelos + 2, 6))
    for j, (rotulo, cor) in enumerate(REGRAS):
        # desloca cada regra para dentro do grupo, centralizado em x
        offset = (j - (n_regras - 1) / 2) * largura
        barras = ax.bar(x + offset, f1_matriz[:, j], largura,
                        label=rotulo, color=cor, edgecolor="white", linewidth=0.8)
        ax.bar_label(barras, fmt="%.1f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 da classe phishing (%)")
    ax.set_title("F1 por modelo nas três regras operacionais")
    ax.legend(title="Regra", loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "f1_barras_comparacao.png")
    fig.savefig(out, dpi=150)
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    # usa a mesma selecao/catalogo de plot.py
    from plot import MODELOS, MODELOS_DISPONIVEIS
    modelos = [MODELOS_DISPONIVEIS[a] for a in MODELOS if a in MODELOS_DISPONIVEIS]
    gerar(modelos, OUT_PATH, show=True)


if __name__ == "__main__":
    main()
