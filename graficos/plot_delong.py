"""
Gera UM grafico do teste de DeLong comparando a AUC de cada modelo LOCAL com a
do Claude (o baseline). O teste de DeLong avalia se a diferenca entre duas AUCs
(areas sob a ROC) e estatisticamente significativa, levando em conta a
covariancia entre as duas curvas por serem avaliadas SOBRE OS MESMOS e-mails
(teste pareado/correlacionado, bem mais sensivel que comparar ICs isolados).
Como a AUC nao depende de limiar, comparamos os scores brutos
(phishing_likelihood) diretamente -- nenhum limiar e aplicado aqui.

Usa a biblioteca MLStatKit (MLstatkit.Delong_test), que devolve a estatistica z
(com sinal = AUC_local - AUC_Claude), o p-valor bicaudal e o IC95% de cada AUC.

Cada par mostra a AUC do Claude e a do modelo local com seus IC95%; o p-valor e
a significancia (estrelas) do teste de DeLong aparecem sobre o par. A comparacao
e feita sobre o conjunto de e-mails validos COMUM aos dois modelos (alinhados
por email_id), com o MESMO rotulo verdadeiro.

O catalogo de modelos vem de plot.py (MODELOS_DISPONIVEIS / MODELOS). Rode
`python plot_delong.py` a partir de qualquer diretorio. A figura e salva em
graficos/images/delong.png.
"""

# ============================================================
BASELINE = "Claude"    # apelido do modelo usado como referencia
ALPHA = 0.95           # nivel de confianca dos ICs de AUC
SEED = 42              # semente (so usada no fallback de bootstrap do DeLong)
OUT_PATH = None        # None = graficos/images/delong.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLstatkit import Delong_test

from limiar import carregar
from plot import MODELOS_DISPONIVEIS, MODELOS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

COR_BASE = "#185FA5"   # azul    -> AUC do Claude (baseline)
COR_LOCAL = "#D85A30"  # laranja -> AUC do modelo local


def _carregar_modelo(log_path):
    """Le o log e devolve (y, score) indexados por email_id (so linhas validas).

    y      -> rotulo verdadeiro (0/1)
    score  -> phishing_likelihood (0-100), usado direto na AUC (sem limiar).
    """
    df = pd.read_csv(os.path.join(ROOT, log_path))
    y_all, score_all, valid = carregar(df)
    ids = df.loc[valid, "email_id"].values
    y = pd.Series(y_all[valid].astype(int).values, index=ids)
    score = pd.Series(score_all[valid].astype(float).values, index=ids)
    return y, score


def _significancia(p):
    """Estrelas convencionais a partir do p-valor."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def gerar(baseline=BASELINE, modelos=MODELOS, alpha=ALPHA, seed=SEED,
          out_path=None, show=False):
    if baseline not in MODELOS_DISPONIVEIS:
        print(f"[erro] baseline desconhecido: {baseline}")
        return None

    nome_base, log_base = MODELOS_DISPONIVEIS[baseline]
    try:
        y_base, s_base = _carregar_modelo(log_base)
    except FileNotFoundError:
        print(f"[erro] log do baseline nao encontrado: {log_base}")
        return None

    # resultados: (nome, auc_loc, ci_loc, auc_base, ci_base, z, p, sig, n)
    resultados = []
    for apelido in modelos:
        if apelido == baseline:
            continue
        if apelido not in MODELOS_DISPONIVEIS:
            print(f"[aviso] modelo desconhecido, pulando: {apelido}")
            continue
        nome, log_path = MODELOS_DISPONIVEIS[apelido]
        try:
            y_loc, s_loc = _carregar_modelo(log_path)
        except FileNotFoundError:
            print(f"[aviso] log nao encontrado, pulando: {log_path}")
            continue

        # alinha pelos email_id comuns e validos nos dois modelos
        comum = y_base.index.intersection(y_loc.index)
        y = y_base.loc[comum].values                  # rotulo verdadeiro (igual nos dois)
        sc = s_base.loc[comum].values                 # A = Claude
        sl = s_loc.loc[comum].values                  # B = modelo local

        if len(np.unique(y)) < 2:
            print(f"[aviso] uma so classe no conjunto comum, pulando: {nome}")
            continue

        # DeLong: z tem sinal AUC_B - AUC_A = AUC_local - AUC_Claude
        z, p, ci_base, ci_loc, auc_base, auc_loc, info = Delong_test(
            y, sc, sl, alpha=alpha, random_state=seed)
        sig = _significancia(p)
        resultados.append((nome, auc_loc, ci_loc, auc_base, ci_base, z, p, sig, len(comum)))

        metodo = info.get("method", "?")
        print(f"{nome} vs {nome_base}: n={len(comum)}  "
              f"AUC_local={auc_loc:.4f} IC=[{ci_loc[0]:.4f}, {ci_loc[1]:.4f}]  "
              f"AUC_{nome_base}={auc_base:.4f} IC=[{ci_base[0]:.4f}, {ci_base[1]:.4f}]  "
              f"dAUC={auc_loc - auc_base:+.4f}  z={z:+.2f}  p={p:.3g}  {sig}  ({metodo})")

    if not resultados:
        print("[erro] nenhum modelo local valido para comparar.")
        return None

    out = _plot(resultados, nome_base, alpha, out_path, show)
    return out


def _plot(resultados, nome_base, alpha, out_path, show):
    """AUC do Claude vs cada modelo local, com IC e p-valor do DeLong por par."""
    nomes = [r[0] for r in resultados]
    x = np.arange(len(nomes))
    desloc = 0.16   # afastamento dos dois pontos dentro do par

    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(nomes) + 2), 6.5))

    for i, (_, auc_loc, ci_loc, auc_base, ci_base, _, _, _, _) in enumerate(resultados):
        # baseline (Claude) a esquerda, modelo local a direita
        ax.errorbar(x[i] - desloc, auc_base,
                    yerr=[[auc_base - ci_base[0]], [ci_base[1] - auc_base]],
                    fmt="o", color=COR_BASE, ecolor=COR_BASE,
                    elinewidth=2.2, capsize=6, capthick=2.2, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5,
                    label=f"AUC {nome_base}" if i == 0 else None)
        ax.errorbar(x[i] + desloc, auc_loc,
                    yerr=[[auc_loc - ci_loc[0]], [ci_loc[1] - auc_loc]],
                    fmt="s", color=COR_LOCAL, ecolor=COR_LOCAL,
                    elinewidth=2.2, capsize=6, capthick=2.2, markersize=10,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=5,
                    label="AUC modelo local" if i == 0 else None)
        ax.text(x[i] - desloc, ci_base[1] + 0.006, f"{auc_base:.3f}",
                ha="center", va="bottom", fontsize=9, color=COR_BASE)
        ax.text(x[i] + desloc, ci_loc[1] + 0.006, f"{auc_loc:.3f}",
                ha="center", va="bottom", fontsize=9, color=COR_LOCAL)

    # limites do eixo y a partir dos ICs, com folga para as anotacoes
    los = [r[2][0] for r in resultados] + [r[4][0] for r in resultados]
    his = [r[2][1] for r in resultados] + [r[4][1] for r in resultados]
    lo_all, hi_all = min(los), max(his)
    span = max(hi_all - lo_all, 1e-3)
    y_topo = hi_all + span * 0.22
    ax.set_ylim(lo_all - span * 0.10, y_topo + span * 0.30)

    # p-valor + significancia do DeLong sobre cada par
    for i, (_, _, _, _, _, z, p, sig, _) in enumerate(resultados):
        ax.text(x[i], y_topo,
                f"DeLong\nz={z:+.1f}\np={p:.2g} {sig}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # linha do acaso (AUC = 0.5), so aparece se estiver dentro da faixa
    if lo_all - span * 0.10 <= 0.5 <= y_topo:
        ax.axhline(0.5, color="#777", ls=":", lw=1.0, zorder=1)
        ax.text(ax.get_xlim()[1], 0.5, " acaso (0,5)", va="center", ha="left",
                fontsize=8, color="#777")

    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylabel(f"AUC (area sob a ROC) -- barras = IC {int(alpha * 100)}%")
    ax.legend(loc="center left", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.13,
            "ns: p$\\geq$0,05    *: p<0,05    **: p<0,01    ***: p<0,001"
            f"   (DeLong bicaudal, pareado nos mesmos e-mails; IC {int(alpha * 100)}%)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "delong.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    gerar(BASELINE, MODELOS, ALPHA, SEED, OUT_PATH, show=True)


if __name__ == "__main__":
    main()
