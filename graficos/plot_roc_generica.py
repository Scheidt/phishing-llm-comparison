"""
Gera uma curva ROC *generica* e didatica para a fundamentacao teorica.

Nao representa nenhum modelo real: a curva e sintetizada a partir de duas
distribuicoes de score (classe positiva x negativa) apenas para ilustrar a
forma tipica de uma boa curva ROC, a diagonal do classificador aleatorio,
o canto ideal (0,1) e o significado dos eixos.

Uso: `python plot_roc_generica.py`
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")


def curva_roc_sintetica():
    """Curva ROC suave e generica (forma tipica de bom classificador)."""
    # FPR de 0 a 1; TPR como funcao concava acima da diagonal.
    # TPR = FPR^p com p<1 gera a "barriga" caracteristica para cima.
    fpr = np.linspace(0.0, 1.0, 500)
    tpr = fpr ** 0.32
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def gerar(out_path=None, show=False):
    fpr, tpr, auc = curva_roc_sintetica()

    fig, ax = plt.subplots(figsize=(7, 7))

    # curva ROC generica
    ax.plot(fpr, tpr, color="#185FA5", lw=2.5,
            label=f"Curva ROC (AUC = {auc:.2f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#185FA5")

    # diagonal do classificador aleatorio (AUC = 0.5)
    ax.plot([0, 1], [0, 1], color="#B4B2A9", ls="--", lw=1.5,
            label="Classificador aleatorio (AUC = 0,50)")

    # curva ROC do classificador perfeito: sobe de (0,0) ate (0,1) e
    # segue ate (1,1) -- as bordas esquerda e superior do grafico
    ax.plot([0, 0, 1], [0, 1, 1], color="#0F6E56", lw=2.0, ls=(0, (5, 2)),
            label="Classificador perfeito (AUC = 1,00)")
    ax.scatter([0], [1], s=130, color="#0F6E56", marker="*",
               edgecolors="white", linewidths=1.0, zorder=6)

    # um ponto operacional generico sobre a curva, so para ilustrar o trade-off
    fp_op = 0.18
    tp_op = fp_op ** 0.32
    ax.scatter([fp_op], [tp_op], s=110, color="#D85A30", marker="o",
               edgecolors="white", linewidths=1.2, zorder=5,
               label="Limiar de decisao (ponto de operacao)")

    # anotacoes didaticas
    ax.annotate("melhor desempenho\n(canto superior esquerdo)",
                xy=(0.02, 0.985), xytext=(0.12, 0.84),
                fontsize=9, color="#0F6E56",
                arrowprops=dict(arrowstyle="->", color="#0F6E56", lw=1.2))
    ax.annotate("cada ponto da curva\ncorresponde a um limiar",
                xy=(fp_op, tp_op), xytext=(0.26, 0.46),
                fontsize=9, color="#D85A30",
                arrowprops=dict(arrowstyle="->", color="#D85A30", lw=1.2))
    ax.text(0.74, 0.70, "acaso", rotation=33, fontsize=9,
            color="#8A887F", ha="center", va="center")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Taxa de falsos positivos  (1 - especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos  (Revocação / recall)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "roc_generica.png")
    fig.savefig(out, dpi=150)
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    gerar(show=True)


if __name__ == "__main__":
    main()
