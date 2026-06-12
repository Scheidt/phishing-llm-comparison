"""
Experimento de validacao do LIMIAR OTIMO com particao treino/teste (holdout).

O limiar otimo de F1 escolhido sobre TODO o conjunto e otimista: ele e ajustado
nos mesmos dados em que e reportado (overfitting de selecao de limiar). Aqui o
limiar e fittado APENAS no treino (70%) e a metrica e reportada no teste (30%),
que o limiar nunca viu. A particao e ESTRATIFICADA por rotulo (mantem a proporcao
phishing/legitimo nos dois lados).

Gera duas figuras:

  1. Curva F1 x limiar com treino e teste SOBREPOSTOS, por modelo
     (holdout_curva_<modelo>.png). Uma particao representativa (SEED). A linha
     vertical marca t* = argmax(F1_treino) -- o limiar efetivamente escolhido.
     Le-se nela o fenomeno do overfitting: a curva de TESTE em t* costuma estar
     abaixo do seu proprio pico; essa distancia vertical e a penalidade.

  2. Barras agrupadas comparando os modelos (holdout_barras.png), com tres
     barras por modelo, agregadas sobre N_SPLITS particoes estratificadas
     (media +- IC percentil):
       - Treino @ t*        (in-sample, otimista)
       - Teste  @ t*        (out-of-sample; o valor de referencia)
       - Teste  @ otimo-teste (teto inatingivel: limiar ajustado no proprio teste)
     A queda da 1a para a 2a barra e o otimismo do limiar; a distancia da 2a para
     a 3a e o custo de ter escolhido o limiar fora do teste.

O catalogo de modelos vem de plot.py (MODELOS_DISPONIVEIS / MODELOS). Execute
`python plot_holdout.py` a partir de qualquer diretorio.
"""

# ============================================================
TEST_SIZE = 0.30       # fracao do holdout de teste (30%)
N_SPLITS = 300         # particoes estratificadas para o grafico de barras (media +- IC)
CONF = 0.95            # nivel de confianca do IC das barras
SEED = 42              # semente: particao representativa da curva + base das repeticoes
STEP = 0.5             # resolucao da varredura do limiar (0..100) na curva
OUT_CURVA = None       # None = images/holdout_curva_<modelo>.png
OUT_BARRAS = None      # None = images/holdout_barras.png
# ============================================================

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from limiar import f1_phishing, melhor_limiar_f1, carregar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)          # raiz do projeto
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")

COR_TREINO = "#185FA5"   # azul  -> curva/bar a de treino (in-sample)
COR_TESTE = "#0F6E56"    # verde -> curva/barra de teste (out-of-sample)
COR_TETO = "#B4B2A9"     # cinza -> teto (limiar otimo do proprio teste)


def _carregar_valido(log_path):
    """Le o log e devolve (y, score) numericos so das linhas validas."""
    df = pd.read_csv(os.path.join(ROOT, log_path))
    y_all, score_all, valid = carregar(df)
    y = y_all[valid].astype(int).values
    score = score_all[valid].astype(float).values
    return y, score


def _split_estratificado(y, score, test_size, seed):
    """Particao estratificada por rotulo. Devolve (y_tr, s_tr, y_te, s_te)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return y[tr], score[tr], y[te], score[te]


# --------------------------------------------------------------------------
# Figura 1: curva F1 x limiar (treino vs teste) de uma particao representativa
# --------------------------------------------------------------------------
def gerar_curva(model_name, log_path, test_size=TEST_SIZE, seed=SEED,
                step=STEP, out_path=None, show=False):
    y, score = _carregar_valido(log_path)
    y_tr, s_tr, y_te, s_te = _split_estratificado(y, score, test_size, seed)

    # curvas suaves de F1 x limiar para treino e teste
    thresholds = np.arange(0.0, 100.0 + step, step)
    f1_tr = np.array([f1_phishing(y_tr, (s_tr >= t).astype(int)) for t in thresholds])
    f1_te = np.array([f1_phishing(y_te, (s_te >= t).astype(int)) for t in thresholds])

    # limiar escolhido NO TREINO (o que o experimento usa de verdade)
    t_star, f1_tr_star = melhor_limiar_f1(y_tr, s_tr)
    # F1 do teste NESSE limiar (out-of-sample) e o teto do teste (limiar otimo do teste)
    f1_te_em_star = f1_phishing(y_te, (s_te >= t_star).astype(int))
    t_te_opt, f1_te_opt = melhor_limiar_f1(y_te, s_te)

    print(f"\n{model_name}  (treino n={len(y_tr)}, teste n={len(y_te)})")
    print(f"  t* (treino)        = {t_star:g}   F1_treino = {f1_tr_star*100:.1f}%")
    print(f"  F1_teste @ t*       = {f1_te_em_star*100:.1f}%   (out-of-sample)")
    print(f"  F1_teste @ otimo    = {f1_te_opt*100:.1f}%  (t={t_te_opt:g}, teto)")
    print(f"  otimismo (queda)    = {(f1_tr_star - f1_te_em_star)*100:+.1f} pp")

    tr_pct = round((1 - test_size) * 100)
    te_pct = round(test_size * 100)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_tr * 100, color=COR_TREINO, lw=2,
            label=f"Treino ({tr_pct}%, n={len(y_tr)})")
    ax.plot(thresholds, f1_te * 100, color=COR_TESTE, lw=2,
            label=f"Teste ({te_pct}%, n={len(y_te)})")
    ax.axvline(t_star, color="#8A8780", ls="--", lw=1.2)

    # ponto do limiar escolhido no treino e como ele cai no teste
    ax.scatter([t_star], [f1_tr_star * 100], s=120, color=COR_TREINO,
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.scatter([t_star], [f1_te_em_star * 100], s=120, color=COR_TESTE,
               edgecolors="white", linewidths=1.2, zorder=5)
    # teto do teste (seu proprio otimo), em contorno para distinguir
    ax.scatter([t_te_opt], [f1_te_opt * 100], s=120, facecolors="none",
               edgecolors=COR_TESTE, linewidths=1.8, zorder=5,
               label=f"otimo do teste (F1={f1_te_opt*100:.1f}%)")

    # seta da "penalidade de generalizacao" em t*
    ax.annotate("", xy=(t_star, f1_te_em_star * 100), xytext=(t_star, f1_tr_star * 100),
                arrowprops=dict(arrowstyle="<->", color="#992D2D", lw=1.5))
    ax.annotate(f"otimismo\n{(f1_tr_star - f1_te_em_star)*100:+.1f} pp",
                xy=(t_star, (f1_tr_star + f1_te_em_star) * 50),
                xytext=(10, 0), textcoords="offset points",
                fontsize=9, color="#992D2D", va="center")
    ax.annotate(f"t* = {t_star:g}", xy=(t_star, 3), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color="#8A8780")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Limiar sobre phishing_likelihood")
    ax.set_ylabel("F1 da classe phishing (%)")
    ax.legend(loc="lower center", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(
        IMAGES_DIR, f"holdout_curva_{re.sub(r'[^A-Za-z0-9._-]+', '_', model_name)}.png")
    fig.savefig(out, dpi=150)
    print(f"  figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Figura 2: barras agrupadas (media +- IC sobre N_SPLITS particoes)
# --------------------------------------------------------------------------
def _repetir_splits(y, score, test_size, n_splits, seed):
    """Sobre n_splits particoes estratificadas, devolve 3 arrays (n_splits,):

    f1_tr_star  -> F1 do treino no limiar otimo do treino (in-sample)
    f1_te_star  -> F1 do teste nesse mesmo limiar (out-of-sample)
    f1_te_opt   -> F1 do teste no limiar otimo do proprio teste (teto)
    """
    rng = np.random.default_rng(seed)
    a = np.empty(n_splits)
    b = np.empty(n_splits)
    c = np.empty(n_splits)
    for i in range(n_splits):
        s = int(rng.integers(0, 2**31 - 1))
        y_tr, s_tr, y_te, s_te = _split_estratificado(y, score, test_size, s)
        t_star, f1_tr_star = melhor_limiar_f1(y_tr, s_tr)
        a[i] = f1_tr_star
        b[i] = f1_phishing(y_te, (s_te >= t_star).astype(int))
        c[i] = melhor_limiar_f1(y_te, s_te)[1]
    return a, b, c


def _media_ic(valores, conf):
    """(media, lo, hi) com IC percentil das repeticoes."""
    alpha = (1 - conf) / 2
    return (float(np.mean(valores)),
            float(np.percentile(valores, 100 * alpha)),
            float(np.percentile(valores, 100 * (1 - alpha))))


def gerar_barras(modelos, test_size=TEST_SIZE, n_splits=N_SPLITS, conf=CONF,
                 seed=SEED, out_path=None, show=False):
    """modelos: lista de (nome_exibicao, log_path)."""
    grupos = [
        ("Treino @ t*", COR_TREINO),
        ("Teste @ t*", COR_TESTE),
        ("Teste @ otimo (teto)", COR_TETO),
    ]
    nomes = []
    medias = []   # por modelo: [(m,lo,hi) treino, (m,lo,hi) teste, (m,lo,hi) teto]
    n_tr_rep = n_te_rep = None   # tamanhos representativos do split (1o modelo valido)
    for nome, log_path in modelos:
        try:
            y, score = _carregar_valido(log_path)
        except FileNotFoundError:
            print(f"[aviso] log nao encontrado, pulando: {log_path}")
            continue
        if n_te_rep is None:
            n_te_rep = round(test_size * len(y))
            n_tr_rep = len(y) - n_te_rep
        a, b, c = _repetir_splits(y, score, test_size, n_splits, seed)
        linha = [_media_ic(a, conf), _media_ic(b, conf), _media_ic(c, conf)]
        nomes.append(nome)
        medias.append(linha)
        print(f"{nome}: treino@t*={linha[0][0]*100:.1f}  "
              f"teste@t*={linha[1][0]*100:.1f}  teste@otimo={linha[2][0]*100:.1f}  "
              f"otimismo={(linha[0][0]-linha[1][0])*100:+.1f}pp")

    if not nomes:
        print("[erro] nenhum modelo valido.")
        return None

    n_mod = len(nomes)
    n_g = len(grupos)
    x = np.arange(n_mod)
    largura = 0.8 / n_g

    # rotulos da legenda com o split: percentual + n real (ex.: "70%, ~1400")
    tr_pct = round((1 - test_size) * 100)
    te_pct = round(test_size * 100)
    rotulos = [
        f"Treino @ t* ({tr_pct}%, ~{n_tr_rep})",
        f"Teste @ t* ({te_pct}%, ~{n_te_rep})",
        "Teste @ otimo (teto)",
    ]

    fig, ax = plt.subplots(figsize=(2.8 * n_mod + 2, 6))
    for j, (_, cor) in enumerate(grupos):
        offset = (j - (n_g - 1) / 2) * largura
        alturas = np.array([medias[i][j][0] * 100 for i in range(n_mod)])
        lo = np.array([medias[i][j][1] * 100 for i in range(n_mod)])
        hi = np.array([medias[i][j][2] * 100 for i in range(n_mod)])
        yerr = np.vstack([alturas - lo, hi - alturas])
        barras = ax.bar(x + offset, alturas, largura, label=rotulos[j], color=cor,
                        edgecolor="white", linewidth=0.8,
                        yerr=yerr, capsize=4, error_kw=dict(elinewidth=1.2))
        ax.bar_label(barras, fmt="%.1f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(nomes)
    ax.set_ylim(0, 108)
    ax.set_ylabel("F1 da classe phishing (%)")
    ax.legend(title="Regra", loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(IMAGES_DIR, "holdout_barras.png")
    fig.savefig(out, dpi=150)
    print(f"Figura salva em: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out


def main():
    from plot import MODELOS, MODELOS_DISPONIVEIS
    modelos = [MODELOS_DISPONIVEIS[a] for a in MODELOS if a in MODELOS_DISPONIVEIS]

    # uma curva por modelo (particao representativa)
    for nome, log_path in modelos:
        try:
            gerar_curva(nome, log_path)
        except FileNotFoundError:
            print(f"[aviso] log nao encontrado, pulando: {log_path}")

    # barras comparativas (todas as repeticoes)
    print("\n########## barras comparativas ##########")
    gerar_barras(modelos, show=True)


if __name__ == "__main__":
    main()
