"""
Funcoes compartilhadas para o limiar otimo sobre phishing_likelihood.

O limiar otimo e o MENOR valor que maximiza o F1 da classe phishing. Como a
predicao (score >= limiar) so muda quando o limiar cruza um score observado,
o F1 e constante entre dois scores consecutivos (um plato): o menor limiar
desse plato otimo e bem definido. Usado por plot_roc.py, plot_limiar.py e
plot_matriz_confusao.py para que os tres concordem no mesmo ponto.
"""

import numpy as np
import pandas as pd

COL_TRUE = "true_label"            # 0 = legitimo, 1 = phishing
COL_SCORE = "phishing_likelihood"  # 0-100
COL_PRED = "predicted_label"       # usado so para descartar -1 (falha de parse)


def f1_phishing(y_true, y_pred):
    """F1 da classe phishing (1 = positivo)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def melhor_limiar_f1(y_true, score):
    """Menor limiar que maximiza o F1 de phishing (regra: score >= limiar).

    O F1 e constante entre dois scores observados consecutivos (a predicao so
    muda quando o limiar cruza um score), entao o otimo costuma ser um plato.
    Os candidatos sao os pontos medios entre scores consecutivos (a fronteira
    de decisao convencional, independente da grade), mais o menor score (tudo
    classificado como phishing). Varrendo em ordem crescente com '>' estrito,
    em caso de empate fica o MENOR limiar do plato otimo. Retorna (limiar, f1).
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    us = np.unique(np.round(score, 4))
    if len(us) == 0:
        return 0.0, 0.0
    # menor score (tudo positivo) + pontos medios entre scores consecutivos
    cands = [float(us[0])] + [float((us[i] + us[i + 1]) / 2) for i in range(len(us) - 1)]
    best_t, best_f1 = cands[0], -1.0
    for t in cands:
        f1 = f1_phishing(y_true, (score >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def carregar(df):
    """Filtro de linhas validas comum aos tres graficos.

    Devolve (y, score, valid): as Series numericas de rotulo verdadeiro e de
    score, mais a mascara booleana das linhas validas (score e rotulo
    numericos e, se a coluna existir, predicao != -1 -> sem falha de parse).
    Cada script aplica a mascara como precisar.
    """
    score = pd.to_numeric(df[COL_SCORE], errors="coerce")
    y = pd.to_numeric(df[COL_TRUE], errors="coerce")
    valid = score.notna() & y.isin([0, 1])
    if COL_PRED in df.columns:
        valid &= df[COL_PRED] != -1
    return y, score, valid
