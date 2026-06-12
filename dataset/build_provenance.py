"""
Tabela de proveniência do dataset do benchmark (para o relatório de reprodutibilidade).

Para CADA e-mail do dataset final efetivamente usado (dataset/emails_dataset.csv),
resolve sua origem no CSV bruto:

  - LEGÍTIMOS (label=0): origem única, dataset/legitimo/emails.csv (corpus
    Enron). O identificador original é o campo `file` do Enron (ex.:
    "kaminski-v/conferences/133.").
  - PHISHING (label=1): origem em UM dos três CSVs, resolvida pela MESMA regra
    de prioridade do pipeline (phishing.csv > phishing_2024.csv > Nazario_5.csv;
    dedup por corpo mantendo a primeira ocorrência). Como esses CSVs não têm
    coluna de id, o identificador original é o índice da linha de dados (0-based,
    igual ao índice do pandas após o cabeçalho).

A fonte de verdade é o próprio dataset final (o que foi dado aos modelos), e
não o dataset_sample_ids.json, que pode estar defasado. O casamento
dataset->origem é feito por CORPO NORMALIZADO (espaços colapsados), porque o
round-trip de to_csv/read_csv altera quebras de linha dentro do corpo de ~5%
dos phishing, quebrando o hash exato embora o e-mail seja o mesmo.

Saída: dataset/dataset_provenance.csv com colunas
    dataset_id, label, src_id, source_file, source_row
e um resumo por CSV de origem no stdout.

Uso:
    python dataset/build_provenance.py
"""
import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_dataset import (  # noqa: E402
    BASE_PHISHING_FILES,
    FILL_PHISHING_FILES,
    body_is_semantic,
    content_key,
    iter_enron_rows,
    _parse_enron_message,
)

csv.field_size_limit(2**31 - 1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "emails_dataset.csv")
LEGIT_CSV = os.path.join(HERE, "legitimo", "emails.csv")
PHISHING_DIR = os.path.join(HERE, "phishing")
PROVENANCE_OUT = os.path.join(HERE, "dataset_provenance.csv")


def _norm(body: str) -> str:
    """Chave de corpo robusta a round-trip de CSV: colapsa todo espaço em branco."""
    return " ".join(str(body).split())


def phishing_origin_by_body(folder: str) -> dict[str, tuple[str, int, str]]:
    """Mapeia corpo_normalizado -> (csv_de_origem, linha_0based, src_id_do_source).

    Replica a prioridade/dedup do pipeline: percorre BASE (phishing.csv,
    phishing_2024.csv) e depois FILL (Nazario_5.csv), filtra label==1 + corpo
    semântico, e mantém a PRIMEIRA ocorrência de cada corpo.
    """
    origin: dict[str, tuple[str, int, str]] = {}
    for name in BASE_PHISHING_FILES + FILL_PHISHING_FILES:
        df = pd.read_csv(os.path.join(folder, name))
        df = df[df["label"] == 1]
        for row_idx, subj, body in zip(
            df.index, df["subject"].fillna("").astype(str), df["body"].fillna("").astype(str)
        ):
            if not body_is_semantic(body):
                continue
            key = _norm(body)
            if key in origin:
                continue  # mantém-se a 1ª ocorrência (prioridade do CSV)
            origin[key] = (name, int(row_idx), content_key(subj, body))
    return origin


def legit_origin_by_body(path: str, wanted_bodies: set[str]) -> dict[str, str]:
    """Mapeia corpo_normalizado -> `file` do Enron, varrendo o corpus uma única vez.

    Só parseia até achar todos os corpos pedidos (os 1000 legítimos do dataset).
    """
    found: dict[str, str] = {}
    for file, raw in iter_enron_rows(path):
        _, _, body = _parse_enron_message(raw)
        key = _norm(body)
        if key in wanted_bodies and key not in found:
            found[key] = file
            if len(found) == len(wanted_bodies):
                break
    return found


def main() -> None:
    ds = pd.read_csv(DATASET_PATH)
    ds["body"] = ds["body"].fillna("").astype(str)
    ds["subject"] = ds["subject"].fillna("").astype(str)

    print("Resolvendo origem dos phishing...", file=sys.stderr)
    ph_origin = phishing_origin_by_body(PHISHING_DIR)

    legit_bodies = {_norm(b) for b, lab in zip(ds["body"], ds["label"]) if lab == 0}
    print(f"Varrendo o Enron para resolver {len(legit_bodies)} legítimos (pode demorar)...", file=sys.stderr)
    legit_origin = legit_origin_by_body(LEGIT_CSV, legit_bodies)

    rows, unresolved = [], 0
    for _, r in ds.iterrows():
        key = _norm(r["body"])
        if r["label"] == 1:
            hit = ph_origin.get(key)
            if hit is None:
                unresolved += 1
                source_file, source_row, src_id = "?", "", content_key(r["subject"], r["body"])
            else:
                name, src_row, src_id = hit
                source_file, source_row = f"dataset/phishing/{name}", src_row
        else:
            file = legit_origin.get(key)
            if file is None:
                unresolved += 1
            source_file = "dataset/legitimo/emails.csv"
            source_row = ""  # o id original do Enron É o src_id (campo `file`)
            src_id = file if file is not None else "?"
        rows.append(
            {
                "dataset_id": int(r["id"]),
                "label": int(r["label"]),
                "src_id": src_id,
                "source_file": source_file,
                "source_row": source_row,
            }
        )

    out = pd.DataFrame(rows, columns=["dataset_id", "label", "src_id", "source_file", "source_row"])
    out.to_csv(PROVENANCE_OUT, index=False)

    print(f"\nProveniência salva em {PROVENANCE_OUT} ({len(out)} amostras)")
    if unresolved:
        print(f"AVISO: {unresolved} amostras não resolvidas (marcadas com '?').")
    print("\nOrigem dos PHISHING (label=1):")
    ph = out[out["label"] == 1]
    for src, n in ph["source_file"].value_counts().items():
        print(f"  {src:40s} {n:4d}")
    print("\nOrigem dos LEGÍTIMOS (label=0):")
    for src, n in out[out["label"] == 0]["source_file"].value_counts().items():
        print(f"  {src:40s} {n:4d}")


if __name__ == "__main__":
    main()
