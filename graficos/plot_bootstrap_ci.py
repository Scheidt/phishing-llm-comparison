"""
Intervalos de confianca (bootstrap PAREADO) da DIFERENCA de metrica entre cada
modelo LOCAL e o Claude (baseline), para tres metricas:

    dF1  = F1(local)  - F1(Claude)     (classe phishing)
    dMCC = MCC(local) - MCC(Claude)    (coef. de correlacao de Matthews)
    dAUC = AUC(local) - AUC(Claude)    (area sob a ROC, independente de limiar)

Como TODOS os modelos sao avaliados sobre OS MESMOS e-mails, o bootstrap e
PAREADO: a cada reamostragem sorteia-se um conjunto de email_id (com reposicao)
e recomputam-se as duas metricas (local e Claude) SOBRE O MESMO conjunto, e so
entao a diferenca. Esse procedimento preserva a correlacao entre os modelos e
da um IC bem mais justo do que tratar cada modelo isoladamente. O conjunto
reamostrado e o dos email_id validos COMUNS ao par (local + Claude), alinhados
por email_id.

F1 e MCC sao avaliados no LIMIAR OTIMO DE F1 de cada modelo (mesmo criterio de
limiar.py / plot_mcnemar.py), calculado UMA vez sobre o conjunto valido completo
do modelo e mantido fixo durante o bootstrap. Com USAR_LIMIAR_PADRAO = True,
todos os modelos usam o mesmo LIMIAR fixo. A AUC nao depende de limiar.

O IC e o intervalo percentil de 95% das diferencas reamostradas; reportam-se
tambem a estimativa pontual (sobre o conjunto comum, sem reamostrar) e um
p-valor de bootstrap bicaudal (fracao de reamostragens que cruza o zero).

O catalogo de modelos vem de plot.py (MODELOS_DISPONIVEIS / MODELOS). Execute
`python plot_bootstrap_ci.py` a partir de qualquer diretorio. A figura e salva
em graficos/images/bootstrap_ci.png.
"""

# ============================================================
USAR_LIMIAR_PADRAO = False  # True = todos os modelos usam o mesmo LIMIAR fixo;
                            # False = limiar otimo de F1 por modelo
LIMIAR = 50.0          # limiar fixo de phishing_likelihood (0-100) usado quando
                       # USAR_LIMIAR_PADRAO = True (score >= LIMIAR -> phishing)
BASELINE = "Claude"    # apelido do modelo usado como referencia
N_BOOT = 10000         # numero de reamostragens de bootstrap
CONF = 0.95            # nivel de confianca do IC
SEED = 42              # semente para reprodutibilidade
OUT_PATH = None        # None = graficos/images/bootstrap_ci.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

from limiar import melhor_limiar_f1, carregar
from plot import MODELOS_DISPONIVEIS, MODELOS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

# ordem e rotulos das tres metricas
METRICAS = [
    ("dF1",  r"$\Delta$F1"),
    ("dMCC", r"$\Delta$MCC"),
    ("dAUC", r"$\Delta$AUC"),
]

COR_POS = "#0F6E56"   # verde  -> IC > 0  (local melhor que o Claude)
COR_NEG = "#992D2D"   # vermelho -> IC < 0 (Claude melhor)
COR_NS  = "#8A8780"   # cinza  -> IC cruza 0 (sem diferenca significativa)


def _carregar_modelo(log_path):
    """Le o log e devolve (y, score, t_opt) indexados por email_id (so validos).

    y      -> rotulo verdadeiro (0/1)
    score  -> phishing_likelihood (0-100)
    t_opt  -> limiar do modelo: o LIMIAR fixo se USAR_LIMIAR_PADRAO, senao o
              limiar otimo de F1, calculado sobre seu conjunto valido completo
              e mantido fixo no bootstrap (igual a plot_mcnemar.py).
    """
    df = pd.read_csv(os.path.join(ROOT, log_path))
    y_all, score_all, valid = carregar(df)
    ids = df.loc[valid, "email_id"].values
    y = pd.Series(y_all[valid].astype(int).values, index=ids)
    score = pd.Series(score_all[valid].astype(float).values, index=ids)
    if USAR_LIMIAR_PADRAO:
        t_opt = LIMIAR
    else:
        t_opt, _ = melhor_limiar_f1(y.values, score.values)
    return y, score, t_opt


def _metricas(y, score, t_opt):
    """(F1, MCC, AUC) da classe phishing para um vetor de score, no limiar t_opt.

    Retorna np.nan em casos degenerados (uma so classe na reamostragem), para
    que o agregado use nanpercentile sem quebrar.
    """
    pred = (score >= t_opt).astype(int)
    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan, np.nan, np.nan
    f1 = f1_score(y, pred, pos_label=1, zero_division=0)
    # MCC e AUC indefinidos se a predicao tem uma so classe / score constante
    mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
    auc = roc_auc_score(y, score) if len(np.unique(score)) > 1 else np.nan
    return f1, mcc, auc


def _bootstrap_par(y, s_loc, s_cla, t_loc, t_cla, n_boot, rng):
    """Bootstrap pareado das diferencas (local - Claude) para as 3 metricas.

    y, s_loc, s_cla sao vetores alinhados (mesmos email_id). A cada iteracao
    sorteiam-se os MESMOS indices para os dois modelos. Devolve um dict
    metrica -> array (n_boot,) das diferencas reamostradas.
    """
    n = len(y)
    out = {"dF1": np.empty(n_boot), "dMCC": np.empty(n_boot), "dAUC": np.empty(n_boot)}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)                # reamostra COM reposicao
        yb = y[idx]
        f1_l, mcc_l, auc_l = _metricas(yb, s_loc[idx], t_loc)
        f1_c, mcc_c, auc_c = _metricas(yb, s_cla[idx], t_cla)
        out["dF1"][b] = f1_l - f1_c
        out["dMCC"][b] = mcc_l - mcc_c
        out["dAUC"][b] = auc_l - auc_c
    return out


def _resumo(diffs, ponto, conf):
    """(estimativa, lo, hi, p_boot, cor) a partir das diferencas reamostradas."""
    alpha = (1 - conf) / 2
    lo = float(np.nanpercentile(diffs, 100 * alpha))
    hi = float(np.nanpercentile(diffs, 100 * (1 - alpha)))
    # p-valor de bootstrap bicaudal: 2x a menor cauda em torno de zero
    n = np.sum(~np.isnan(diffs))
    frac_neg = np.sum(diffs < 0) / n if n else np.nan
    p = float(min(1.0, 2 * min(frac_neg, 1 - frac_neg)))
    if lo > 0:
        cor = COR_POS
    elif hi < 0:
        cor = COR_NEG
    else:
        cor = COR_NS
    return ponto, lo, hi, p, cor


def gerar(baseline=BASELINE, modelos=MODELOS, n_boot=N_BOOT, conf=CONF,
          seed=SEED, out_path=None, show=False):
    if baseline not in MODELOS_DISPONIVEIS:
        print(f"[erro] baseline desconhecido: {baseline}")
        return None

    nome_base, log_base = MODELOS_DISPONIVEIS[baseline]
    try:
        y_base, s_base, t_base = _carregar_modelo(log_base)
    except FileNotFoundError:
        print(f"[erro] log do baseline nao encontrado: {log_base}")
        return None
    print(f"Baseline: {nome_base}  (limiar otimo F1 = {t_base:g})")

    rng = np.random.default_rng(seed)

    # resultados[apelido] = {nome, metrica -> (est, lo, hi, p, cor)}
    resultados = []
    for apelido in modelos:
        if apelido == baseline or apelido not in MODELOS_DISPONIVEIS:
            if apelido != baseline:
                print(f"[aviso] modelo desconhecido, pulando: {apelido}")
            continue
        nome, log_path = MODELOS_DISPONIVEIS[apelido]
        try:
            y_loc, s_loc, t_loc = _carregar_modelo(log_path)
        except FileNotFoundError:
            print(f"[aviso] log nao encontrado, pulando: {log_path}")
            continue

        # alinha pelos email_id comuns e validos nos dois modelos
        comum = y_base.index.intersection(y_loc.index)
        y = y_base.loc[comum].values                       # rotulo verdadeiro (igual nos dois)
        sl = s_loc.loc[comum].values
        sc = s_base.loc[comum].values

        # estimativa pontual (sem reamostrar) e bootstrap pareado
        f1_l, mcc_l, auc_l = _metricas(y, sl, t_loc)
        f1_c, mcc_c, auc_c = _metricas(y, sc, t_base)
        ponto = {"dF1": f1_l - f1_c, "dMCC": mcc_l - mcc_c, "dAUC": auc_l - auc_c}
        diffs = _bootstrap_par(y, sl, sc, t_loc, t_base, n_boot, rng)

        res = {chave: _resumo(diffs[chave], ponto[chave], conf)
               for chave, _ in METRICAS}
        resultados.append((nome, res))

        print(f"\n{nome} vs {nome_base}  (n={len(comum)}, limiar F1 local={t_loc:g})")
        for chave, rot in METRICAS:
            est, lo, hi, p, _ = res[chave]
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  {chave:5s}: {est:+.4f}  IC{int(conf*100)}%=[{lo:+.4f}, {hi:+.4f}]"
                  f"  p_boot={p:.3g}{sig}")

    if not resultados:
        print("[erro] nenhum modelo local valido para comparar.")
        return None

    out = _plot(resultados, nome_base, conf, n_boot, out_path, show)
    return out


def _plot(resultados, nome_base, conf, n_boot, out_path, show):
    """Forest plot: um painel por metrica, modelos no eixo y, IC horizontal."""
    nomes = [r[0] for r in resultados]
    n_mod = len(nomes)
    y_pos = np.arange(n_mod)[::-1]   # primeiro modelo no topo

    fig, axes = plt.subplots(1, len(METRICAS), figsize=(5 * len(METRICAS), 1.3 * n_mod + 2.2),
                             sharey=True)
    if len(METRICAS) == 1:
        axes = [axes]

    for ax, (chave, rotulo) in zip(axes, METRICAS):
        for yp, (nome, res) in zip(y_pos, resultados):
            est, lo, hi, p, cor = res[chave]
            ax.errorbar(est, yp,
                        xerr=[[est - lo], [hi - est]],
                        fmt="o", color=cor, ecolor=cor,
                        elinewidth=2.4, capsize=6, capthick=2.4,
                        markersize=9, markeredgecolor="white", markeredgewidth=1.2,
                        zorder=5)
            # rotulo numerico do IC ao lado do ponto
            ax.text(hi, yp + 0.18,
                    f"{est:+.3f}\n[{lo:+.3f}, {hi:+.3f}]",
                    va="bottom", ha="center", fontsize=8, color=cor)

        ax.axvline(0, color="#333", ls="--", lw=1.2, zorder=1)
        ax.set_title(rotulo, fontsize=13, fontweight="bold")
        ax.set_xlabel(f"Diferenca (local - {nome_base})")
        ax.grid(axis="x", alpha=0.3)
        # margem horizontal folgada para os rotulos
        lo_all = min(res[chave][1] for _, res in resultados)
        hi_all = max(res[chave][2] for _, res in resultados)
        span = max(hi_all - lo_all, 1e-3)
        ax.set_xlim(min(lo_all, 0) - span * 0.18, max(hi_all, 0) + span * 0.30)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(nomes, fontsize=11)
    axes[0].set_ylim(-0.6, n_mod - 0.4)

    # legenda de cores (significancia)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=COR_POS, lw=0, markersize=9,
               label="IC > 0  (local melhor)"),
        Line2D([0], [0], marker="o", color=COR_NEG, lw=0, markersize=9,
               label=f"IC < 0  ({nome_base} melhor)"),
        Line2D([0], [0], marker="o", color=COR_NS, lw=0, markersize=9,
               label="IC cruza 0  (sem diferenca)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Bootstrap pareado das diferencas vs {nome_base}  "
        f"(IC {int(conf*100)}%, {n_boot} reamostragens)",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "bootstrap_ci.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    gerar(BASELINE, MODELOS, N_BOOT, CONF, SEED, OUT_PATH, show=True)


if __name__ == "__main__":
    main()
