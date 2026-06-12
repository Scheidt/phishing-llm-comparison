"""
Gera UM grafico do teste de McNemar comparando cada modelo LOCAL com o Claude
(o baseline). O teste de McNemar avalia se dois classificadores diferem de forma
estatisticamente significativa SOBRE OS MESMOS e-mails, olhando apenas para os
pares DISCORDANTES (um acerta e o outro erra):

    b = Claude acerta  e  modelo local erra
    c = Claude erra     e  modelo local acerta

A estatistica (com correcao de continuidade) e  chi2 = (|b-c|-1)**2 / (b+c),
com 1 grau de liberdade. Para b+c pequeno, o p-valor exato (binomial, p=0.5)
e mais confiavel, entao o binomial exato e usado como p-valor reportado.

Cada barra do par mostra b e c; quanto mais assimetrico o par, mais forte a
evidencia de que um modelo e melhor que o outro. A significancia (estrelas) e o
p-valor aparecem sobre cada par.

A correcao de cada e-mail e recomputada no LIMIAR OTIMO DE F1 de cada modelo
(mesmo criterio de limiar.py usado nos outros graficos), sobre o conjunto de
e-mails validos COMUM aos dois modelos (alinhados por email_id).

O catalogo de modelos vem de plot.py (MODELOS_DISPONIVEIS / MODELOS). Execute
`python plot_mcnemar.py` a partir de qualquer diretorio. A figura e salva em
graficos/images/mcnemar.png.
"""

# ============================================================
BASELINE = "Claude"    # apelido do modelo usado como referencia
OUT_PATH = None        # None = graficos/images/mcnemar.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist, binomtest

from limiar import melhor_limiar_f1, carregar
from plot import MODELOS_DISPONIVEIS, MODELOS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

COR_B = "#185FA5"   # azul   -> Claude acerta, modelo erra
COR_C = "#D85A30"   # laranja -> modelo acerta, Claude erra


def _correto_por_email(log_path):
    """Carrega o log e devolve uma Series 'acertou' (bool) indexada por email_id.

    A predicao e recomputada no limiar otimo de F1 do proprio modelo (mesmo
    criterio dos outros graficos), restrita as linhas validas.
    """
    df = pd.read_csv(os.path.join(ROOT, log_path))
    y_all, score_all, valid = carregar(df)
    y = y_all[valid].astype(int).values
    score = score_all[valid].astype(float).values

    t_opt, _ = melhor_limiar_f1(y, score)
    pred = (score >= t_opt).astype(int)
    acertou = (pred == y)

    ids = df.loc[valid, "email_id"].values
    return pd.Series(acertou, index=ids, name="acertou")


def _mcnemar(b, c):
    """Devolve (chi2_cc, p_exato, significancia) para os discordantes b e c."""
    n = b + c
    if n == 0:
        return 0.0, 1.0, "n/d"
    chi2_cc = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    chi2_cc = max(chi2_cc, 0.0)
    # p-valor exato (binomial bicaudal, H0: p=0.5) -- robusto para n pequeno
    p_exato = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    if p_exato < 0.001:
        sig = "***"
    elif p_exato < 0.01:
        sig = "**"
    elif p_exato < 0.05:
        sig = "*"
    else:
        sig = "ns"
    return chi2_cc, p_exato, sig


def gerar(baseline=BASELINE, modelos=MODELOS, out_path=None, show=False):
    if baseline not in MODELOS_DISPONIVEIS:
        print(f"[erro] baseline desconhecido: {baseline}")
        return None

    nome_base, log_base = MODELOS_DISPONIVEIS[baseline]
    try:
        base_ok = _correto_por_email(log_base)
    except FileNotFoundError:
        print(f"[erro] log do baseline nao encontrado: {log_base}")
        return None

    resultados = []   # (nome, b, c, both_ok, both_wrong, chi2, p, sig)
    for apelido in modelos:
        if apelido == baseline:
            continue
        if apelido not in MODELOS_DISPONIVEIS:
            print(f"[aviso] modelo desconhecido, pulando: {apelido}")
            continue
        nome, log_path = MODELOS_DISPONIVEIS[apelido]
        try:
            mod_ok = _correto_por_email(log_path)
        except FileNotFoundError:
            print(f"[aviso] log nao encontrado, pulando: {log_path}")
            continue

        # alinha pelos email_id comuns e validos nos dois modelos
        comum = base_ok.index.intersection(mod_ok.index)
        cb = base_ok.loc[comum].values
        cm = mod_ok.loc[comum].values

        b = int(np.sum(cb & ~cm))   # Claude acerta, modelo erra
        c = int(np.sum(~cb & cm))   # modelo acerta, Claude erra
        both_ok = int(np.sum(cb & cm))
        both_wrong = int(np.sum(~cb & ~cm))

        chi2_cc, p, sig = _mcnemar(b, c)
        resultados.append((nome, b, c, both_ok, both_wrong, chi2_cc, p, sig))
        print(f"{nome} vs {nome_base}: n={len(comum)}  "
              f"ambos_ok={both_ok}  ambos_erro={both_wrong}  "
              f"b(Claude+/mod-)={b}  c(mod+/Claude-)={c}  "
              f"chi2={chi2_cc:.2f}  p={p:.3g}  {sig}")

    if not resultados:
        print("[erro] nenhum modelo local valido para comparar.")
        return None

    # --- plot: barras pareadas (b, c) por modelo ---
    nomes = [r[0] for r in resultados]
    bs = [r[1] for r in resultados]
    cs = [r[2] for r in resultados]

    x = np.arange(len(nomes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, 2.4 * len(nomes) + 2), 6.5))
    barras_b = ax.bar(x - w / 2, bs, w, color=COR_B,
                      label=f"{nome_base} acerta, modelo erra  (b)")
    barras_c = ax.bar(x + w / 2, cs, w, color=COR_C,
                      label=f"Modelo acerta, {nome_base} erra  (c)")

    for barras in (barras_b, barras_c):
        ax.bar_label(barras, padding=2, fontsize=10)

    # anotacao de p-valor + significancia sobre cada par
    topo = max(max(bs), max(cs)) if (bs or cs) else 1
    for i, (_, b, c, _, _, chi2_cc, p, sig) in enumerate(resultados):
        y = max(b, c) + topo * 0.10
        ax.text(x[i], y,
                f"$\\chi^2$={chi2_cc:.1f}\np={p:.2g} {sig}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, topo * 1.32)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylabel("Numero de e-mails (pares discordantes)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.13,
            "ns: p$\\geq$0,05    *: p<0,05    **: p<0,01    ***: p<0,001"
            "   (p exato, binomial bicaudal)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "mcnemar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    gerar(BASELINE, MODELOS, OUT_PATH, show=True)


if __name__ == "__main__":
    main()
