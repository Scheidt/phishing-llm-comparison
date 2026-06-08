"""
Gera a curva ROC de um modelo a partir do CSV de log, marcando tres pontos operacionais:
  1. classification  -> rotulo textual do modelo (coluna predicted_classification_text)
  2. likelihood>=50   -> corte fixo no meio da escala
  3. F1-otimo         -> corte que maximiza o F1 da classe phishing

Uso: ajuste LOG_PATH abaixo e rode `python plot_roc.py`.
"""

# ============================================================
LOG_PATH = "results/llm_logs/qwen3_4b_20260607_065655.csv"   # <-- caminho do log
MODEL_NAME = "Qwen 3 4B"
OUT_PATH = None        # None = deriva do nome do log (ex.: roc_<modelo>.png)
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

# --- nomes de coluna esperados (true_label/phishing_likelihood/predicted_label
# ficam em limiar.carregar; aqui so os especificos da curva ROC) ---
COL_CLS = "predicted_classification_text"     # "PHISHING" / "LEGITIMO"
COL_RAW = "raw_response"                      # fallback se COL_CLS ausente


def text_to_label(value):
    """Mapeia o rotulo textual do modelo para 1 (phishing) / 0 (legitimo) / None."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if "phish" in v:
        return 1
    if "leg" in v:   # cobre "legitimo" e "legitimate"
        return 0
    return None


def classification_from_raw(raw):
    """Fallback: extrai o campo classification de dentro do JSON da resposta."""
    if not isinstance(raw, str):
        return None
    m = re.search(r'"?classification"?\s*:\s*"?([A-Za-z\u00C0-\u00FF]+)', raw, re.I)
    return text_to_label(m.group(1)) if m else None


def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def rates(y_true, y_pred):
    """Retorna (fpr, tpr, f1) para a classe phishing."""
    tp, fp, tn, fn = confusion(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    return fpr, tpr, f1


def roc_curve_points(y_true, score):
    """Curva ROC computada varrendo todos os limiares distintos do score."""
    thresholds = np.r_[score.max() + 1, np.unique(score)[::-1]]
    pts = []
    for t in thresholds:
        fpr, tpr, _ = rates(y_true, (score >= t).astype(int))
        pts.append((fpr, tpr))
    pts.append((1.0, 1.0))
    pts = sorted(set(pts))
    return np.array(pts)


def auc_trapz(roc):
    x, yv = roc[:, 0], roc[:, 1]
    return float(np.sum((x[1:] - x[:-1]) * (yv[1:] + yv[:-1]) / 2.0))


def gerar(model_name=MODEL_NAME, log_path=LOG_PATH, out_path=None, show=False):
    df = pd.read_csv(os.path.join(ROOT, log_path))

    # rotulo textual do modelo (Regra 1), com fallback para o raw_response
    if COL_CLS in df.columns:
        cls = df[COL_CLS].apply(text_to_label)
    else:
        print(f"[aviso] coluna '{COL_CLS}' ausente; extraindo de '{COL_RAW}'.")
        cls = df[COL_RAW].apply(classification_from_raw)

    # filtro base (score + rotulo validos, sem falha de parse), igual aos
    # outros graficos -> garante o mesmo limiar otimo
    y_all, score_all, valid = carregar(df)
    y = y_all[valid].astype(int).values
    score = score_all[valid].astype(float).values
    print(f"Linhas validas: {valid.sum()} de {len(df)}")

    # ponto da Regra 1 (classification) exige tambem o rotulo textual parseavel
    valid_cls = valid & cls.notna()
    y_cls = y_all[valid_cls].astype(int).values
    cls_v = cls[valid_cls].astype(int).values

    # --- curva ROC e AUC ---
    roc = roc_curve_points(y, score)
    auc = auc_trapz(roc)

    # --- tres pontos operacionais ---
    t_opt, f1_opt = melhor_limiar_f1(y, score)  # menor limiar que maximiza o F1
    p1 = rates(y_cls, cls_v)                     # Regra 1: classification
    p2 = rates(y, (score >= 50).astype(int))    # Regra 2: likelihood>=50
    p3 = rates(y, (score >= t_opt).astype(int)) # Regra 3: F1-otimo

    model = model_name

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(roc[:, 0], roc[:, 1], color="#185FA5", lw=2,
            label=f"Curva ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#B4B2A9", ls="--", lw=1, label="Aleatorio")

    points = [
        ("1 classification",         p1, "#D85A30", "D"),
        ("2 likelihood >= 50",       p2, "#992D2D", "X"),
        (f"3 F1-otimo (t={t_opt:g})", p3, "#0F6E56", "o"),
    ]
    for name, (fpr, tpr, f1), color, marker in points:
        ax.scatter(fpr, tpr, s=120, color=color, marker=marker,
                   edgecolors="white", linewidths=1.2, zorder=5,
                   label=f"{name}  (F1={f1*100:.1f}, FPR={fpr*100:.1f}%, TPR={tpr*100:.1f}%)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Taxa de falsos positivos  (1 - especificidade)")
    ax.set_ylabel("Recall de phishing  (TPR)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = out_path or os.path.join(
        IMAGES_DIR, f"roc_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}.png")
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