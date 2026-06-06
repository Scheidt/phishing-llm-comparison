"""
Análise aprofundada dos logs por e-mail do benchmark.

Vai além do comparison_report.csv (que usa threshold fixo = 50) e responde:
  1. Poder discriminativo real de cada modelo (ROC-AUC), independente do corte.
  2. Threshold ótimo por modelo (maximiza F1) e ganho frente ao corte fixo de 50.
  3. Calibração do phishing_likelihood (Brier score).
  4. Sobreposição de erros entre modelos + potencial de ensemble (voto / média).
  5. Custo: tokens e latência (média, mediana, p90, p95, p99).

Roda offline a partir de results/llm_logs/. Uso: python analysis/deep_analysis.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "results", "llm_logs")
OUT_DIR = os.path.join(ROOT, "results", "reports")
FIXED_THRESHOLD = 50  # config.PHISHING_LIKELIHOOD_THRESHOLD

# Preço por 1M tokens (USD) — só Haiku é pago; locais ~0 de custo monetário direto.
PRICING = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
}


def latest_logs() -> dict[str, str]:
    """Pega o CSV mais recente de cada modelo (prefixo antes do timestamp)."""
    files = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    by_model: dict[str, str] = {}
    for f in sorted(files):
        base = os.path.basename(f)
        # nome = <model>_<YYYYMMDD>_<HHMMSS>.csv
        model = base.rsplit("_", 2)[0]
        by_model[model] = f  # ordenado => fica o mais recente
    return by_model


def load(model: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["model"] = model
    return df


def metrics_at(y_true, score, thr) -> dict:
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main() -> None:
    logs = latest_logs()
    if not logs:
        sys.exit("Nenhum log encontrado em results/llm_logs/")

    frames = {m: load(m, p) for m, p in logs.items()}

    print("=" * 78)
    print("ANÁLISE APROFUNDADA — logs por e-mail")
    print("=" * 78)

    # ---- 1+2+3: ROC-AUC, threshold ótimo, calibração ----
    summary_rows = []
    roc_data = {}
    for model, df in frames.items():
        # Só linhas com likelihood numérico (score disponível).
        d = df.dropna(subset=["phishing_likelihood"]).copy()
        d = d[d["true_label"].isin([0, 1])]
        y = d["true_label"].astype(int).to_numpy()
        score = d["phishing_likelihood"].astype(float).to_numpy()
        n = len(d)

        auc = roc_auc_score(y, score)
        # Brier: likelihood como probabilidade (0-100 -> 0-1).
        brier = brier_score_loss(y, np.clip(score / 100.0, 0, 1))

        fixed = metrics_at(y, score, FIXED_THRESHOLD)

        # Threshold ótimo (varre todos os cortes candidatos, maximiza F1).
        cand = np.unique(score)
        best = max((metrics_at(y, score, t) for t in cand), key=lambda m: m["f1"])

        roc_data[model] = roc_curve(y, score)
        summary_rows.append({
            "model": model, "n_scored": n, "auc": auc, "brier": brier,
            "f1@50": fixed["f1"], "prec@50": fixed["precision"], "rec@50": fixed["recall"],
            "f1_best": best["f1"], "thr_best": best["threshold"],
            "prec_best": best["precision"], "rec_best": best["recall"],
            "fp@50": fixed["fp"], "fp_best": best["fp"],
        })

    summ = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    print("\n[1] PODER DISCRIMINATIVO (independe do threshold)")
    print("    AUC alto = separa bem phishing de legítimo, qualquer que seja o corte.")
    print(summ[["model", "n_scored", "auc", "brier"]].to_string(index=False,
          formatters={"auc": "{:.4f}".format, "brier": "{:.4f}".format}))

    print("\n[2] THRESHOLD FIXO (50) vs THRESHOLD ÓTIMO (max F1) por modelo")
    print(summ[["model", "f1@50", "f1_best", "thr_best", "prec@50", "prec_best",
                "rec@50", "rec_best", "fp@50", "fp_best"]].to_string(index=False,
          formatters={c: "{:.4f}".format for c in
                      ["f1@50", "f1_best", "prec@50", "prec_best", "rec@50", "rec_best"]}))

    # ---- 4: sobreposição de erros + ensemble ----
    print("\n[3] SOBREPOSIÇÃO DE ERROS (no threshold fixo 50)")
    # Alinha por email_id; usa predição no corte 50.
    aligned = None
    for model, df in frames.items():
        d = df.dropna(subset=["phishing_likelihood"]).copy()
        d["pred50"] = (d["phishing_likelihood"].astype(float) >= FIXED_THRESHOLD).astype(int)
        d["wrong"] = (d["pred50"] != d["true_label"]).astype(int)
        sub = d[["email_id", "true_label", "pred50", "wrong"]].rename(
            columns={"pred50": f"pred_{model}", "wrong": f"wrong_{model}"})
        aligned = sub if aligned is None else aligned.merge(
            sub.drop(columns=["true_label"]), on="email_id", how="inner")

    models = list(frames.keys())
    wrong_cols = [f"wrong_{m}" for m in models]
    a = aligned.dropna(subset=wrong_cols).copy()
    n_common = len(a)
    print(f"    e-mails em comum a todos os modelos: {n_common}")
    for m in models:
        print(f"      erros {m:32s}: {int(a[f'wrong_{m}'].sum())}")
    all_wrong = (a[wrong_cols].sum(axis=1) == len(models)).sum()
    none_wrong = (a[wrong_cols].sum(axis=1) == 0).sum()
    print(f"    e-mails que TODOS erram (difíceis de verdade): {all_wrong}")
    print(f"    e-mails que NINGUÉM erra: {none_wrong}")

    # Pareamento de erros: Jaccard de quem erra entre cada par.
    print("\n    Jaccard de erros entre pares (1.0 = erram exatamente os mesmos):")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            wi = a[f"wrong_{models[i]}"].astype(bool)
            wj = a[f"wrong_{models[j]}"].astype(bool)
            inter = (wi & wj).sum()
            union = (wi | wj).sum()
            jac = inter / union if union else 0
            print(f"      {models[i]:28s} x {models[j]:28s}: {jac:.3f}")

    # Ensembles
    print("\n[4] ENSEMBLE (combina os 3 modelos LOCAIS; Haiku fora p/ baseline justo)")
    local = [m for m in models if not m.startswith("claude")]
    if len(local) >= 2:
        yt = a["true_label"].astype(int).to_numpy()
        # Voto majoritário das predições no corte 50.
        votes = np.vstack([a[f"pred_{m}"].to_numpy() for m in local]).sum(axis=0)
        maj = (votes >= (len(local) / 2)).astype(int)
        # Média de likelihood (precisa re-merge dos scores).
        score_df = a[["email_id", "true_label"]].copy()
        for m in local:
            dm = frames[m].dropna(subset=["phishing_likelihood"])[["email_id", "phishing_likelihood"]]
            score_df = score_df.merge(dm.rename(columns={"phishing_likelihood": f"s_{m}"}),
                                      on="email_id", how="left")
        avg_score = score_df[[f"s_{m}" for m in local]].mean(axis=1).to_numpy()

        def show(name, pred):
            print(f"      {name:34s} acc={accuracy_score(yt, pred):.4f} "
                  f"f1={f1_score(yt, pred, zero_division=0):.4f} "
                  f"prec={precision_score(yt, pred, zero_division=0):.4f} "
                  f"rec={recall_score(yt, pred, zero_division=0):.4f}")

        print(f"      (locais: {', '.join(local)})")
        show("voto majoritário @50", maj)
        show("média de likelihood @50", (avg_score >= FIXED_THRESHOLD).astype(int))
        print(f"      AUC ensemble (média de likelihood): {roc_auc_score(yt, avg_score):.4f}")

    # ---- 5: custo / latência ----
    print("\n[5] LATÊNCIA (segundos por e-mail) e CUSTO")
    lat_rows = []
    for model, df in frames.items():
        e = df["elapsed_seconds"].astype(float)
        row = {"model": model, "mean": e.mean(), "p50": e.median(),
               "p90": e.quantile(.90), "p95": e.quantile(.95),
               "p99": e.quantile(.99), "max": e.max()}
        ti = df["input_tokens"].fillna(0).sum()
        to = df["output_tokens"].fillna(0).sum()
        if model in PRICING:
            cost = ti / 1e6 * PRICING[model]["in"] + to / 1e6 * PRICING[model]["out"]
            row["usd_total"] = cost
            row["usd_per_1k"] = cost / len(df) * 1000
        else:
            row["usd_total"] = 0.0
            row["usd_per_1k"] = 0.0
        lat_rows.append(row)
    lat = pd.DataFrame(lat_rows)
    print(lat.to_string(index=False, formatters={
        c: "{:.2f}".format for c in ["mean", "p50", "p90", "p95", "p99", "max"]}
        | {"usd_total": "{:.4f}".format, "usd_per_1k": "{:.4f}".format}))

    # ---- ROC plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 7))
        for model in summ["model"]:
            fpr, tpr, _ = roc_data[model]
            auc = float(summ.loc[summ.model == model, "auc"].iloc[0])
            plt.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=.4)
        plt.xlabel("Taxa de falso positivo (legítimo marcado como phishing)")
        plt.ylabel("Taxa de verdadeiro positivo (recall de phishing)")
        plt.title("Curva ROC — phishing_likelihood como score")
        plt.legend(loc="lower right")
        plt.grid(alpha=.3)
        out_png = os.path.join(OUT_DIR, "roc_curves.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=130)
        print(f"\n[+] Curva ROC salva em: {out_png}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[!] Plot pulado: {exc}")

    # Salva resumo numérico.
    out_csv = os.path.join(OUT_DIR, "deep_analysis_summary.csv")
    summ.to_csv(out_csv, index=False)
    print(f"[+] Resumo salvo em: {out_csv}")


if __name__ == "__main__":
    main()
