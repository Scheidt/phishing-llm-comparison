"""
Preparação reprodutível do dataset CEAS_08 para o benchmark.

ESPECÍFICO DO DATASET CEAS_08 (dataset/CEAS_08.csv). O CEAS_08 tem as
colunas `sender, receiver, date, subject, body, label, urls`; este script
extrai apenas `subject, body, label`, atribui um `id` posicional e produz
o formato esperado pelo pipeline (id, subject, body, label) com uma
amostra aleatória balanceada 50/50 entre phishing (label=1) e legítimos
(label=0).

Estratégia de reprodutibilidade
-------------------------------
Estão salvos os ids das amostras sorteadas (dataset/ceas08_sample_ids.csv) e um
arquivo de metadados (dataset/ceas08_sample_ids.meta.json) com o hash do CSV de origem
e a semente usada. O script pode ser rodado em modo "rebuild" para reconstruir o dataset
a partir dos ids salvos, validando o hash do CSV de origem para garantir que é o mesmo
usado na amostragem.

Isso é mais robusto que confiar só no --seed, pois o resultado de
DataFrame.sample(random_state=...) pode variar entre versões de
pandas/numpy — a lista explícita de ids, não.

Uso:
    python dataset/prepare_ceas08.py                 # sorteia e salva ids + metadados
    python dataset/prepare_ceas08.py --rebuild       # reconstrói o dataset a partir dos ids
    python dataset/prepare_ceas08.py --samples 1000 --seed 7
"""
import argparse
import hashlib
import json
import sys

import pandas as pd

REQUIRED_SOURCE_COLS = {"subject", "body", "label"}


def sha256_file(path: str) -> str:
    """Calcula o SHA-256 de um arquivo em blocos (sem carregá-lo inteiro na memória)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(input_path: str) -> pd.DataFrame:
    """Carrega o CSV de origem, atribui `id` posicional e normaliza/limpa colunas."""
    df = pd.read_csv(input_path)

    missing = REQUIRED_SOURCE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV de origem está faltando colunas: {missing}")

    # id = índice original da linha no CSV de origem (rastreável/reprodutível).
    df = df.reset_index(drop=True)
    df["id"] = df.index

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # Remove e-mails sem conteúdo (subject e body vazios não servem ao benchmark).
    df = df[(df["subject"].str.strip() != "") | (df["body"].str.strip() != "")]
    return df


def write_dataset(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Escreve o dataset no formato esperado pelo pipeline (id, subject, body, label)."""
    out = df[["id", "subject", "body", "label"]]
    out.to_csv(output_path, index=False)
    return out


def sample_dataset(
    input_path: str,
    output_path: str,
    ids_path: str,
    meta_path: str,
    total_samples: int,
    seed: int,
) -> None:
    """Sorteia uma amostra balanceada 50/50 e salva dataset, ids e metadados."""
    df = load_source(input_path)

    per_class = total_samples // 2
    phishing = df[df["label"] == 1]
    legit = df[df["label"] == 0]

    if len(phishing) < per_class or len(legit) < per_class:
        raise ValueError(
            f"Amostras insuficientes para {per_class} por classe "
            f"(disponível: {len(phishing)} phishing, {len(legit)} legítimos). "
            f"Reduza --samples."
        )

    sample_phishing = phishing.sample(n=per_class, random_state=seed)
    sample_legit = legit.sample(n=per_class, random_state=seed)
    sample = pd.concat([sample_phishing, sample_legit])
    # Embaralha a ordem final para não ficar phishing seguidos dos legítimos.
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)

    out = write_dataset(sample, output_path)

    sorted_ids = sorted(out["id"].tolist())
    pd.DataFrame({"id": sorted_ids}).to_csv(ids_path, index=False)

    source_hash = sha256_file(input_path)
    meta = {
        "source_file": input_path,
        "source_sha256": source_hash,
        "total_samples": int(len(out)),
        "per_class": per_class,
        "seed": seed,
        "pandas_version": pd.__version__,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f"Dataset salvo em {output_path}: {len(out)} amostras "
        f"({out['label'].sum()} phishing, {(out['label'] == 0).sum()} legítimos)"
    )
    print(f"Ids salvos em {ids_path} | metadados em {meta_path}")
    print(f"source_sha256={source_hash}")


def rebuild_dataset(
    input_path: str,
    output_path: str,
    ids_path: str,
    meta_path: str,
) -> None:
    """Reconstrói o dataset a partir dos ids salvos, validando o hash do source."""
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    actual_hash = sha256_file(input_path)
    if actual_hash != meta["source_sha256"]:
        sys.exit(
            f"ERRO: o CSV de origem mudou desde a amostragem.\n"
            f"  esperado: {meta['source_sha256']}\n"
            f"  atual:    {actual_hash}\n"
            f"A reconstrução não é confiável — use o mesmo {meta['source_file']} original."
        )

    ids = pd.read_csv(ids_path)["id"].tolist()
    df = load_source(input_path)
    rebuilt = df[df["id"].isin(ids)]

    if len(rebuilt) != len(ids):
        sys.exit(
            f"ERRO: {len(ids)} ids esperados, mas {len(rebuilt)} encontrados no source. "
            f"O arquivo de origem pode não corresponder ao usado na amostragem."
        )

    out = write_dataset(rebuilt, output_path)
    print(
        f"Dataset reconstruído em {output_path}: {len(out)} amostras "
        f"({out['label'].sum()} phishing, {(out['label'] == 0).sum()} legítimos) "
        f"[hash do source validado]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o dataset CEAS_08 para o benchmark (amostra/reconstrói).")
    parser.add_argument("--input", default="dataset/CEAS_08.csv", help="CSV de origem (CEAS_08).")
    parser.add_argument("--output", default="dataset/emails_dataset.csv", help="Dataset de saída.")
    parser.add_argument("--ids", default="dataset/ceas08_sample_ids.csv", help="Arquivo com os ids sorteados.")
    parser.add_argument("--meta", default="dataset/ceas08_sample_ids.meta.json", help="Metadados (hash, seed).")
    parser.add_argument("--samples", type=int, default=2000, help="Total de amostras (dividido 50/50).")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstrói o dataset a partir dos ids/metadados existentes (não re-amostra).",
    )
    args = parser.parse_args()

    if args.rebuild:
        rebuild_dataset(args.input, args.output, args.ids, args.meta)
    else:
        sample_dataset(args.input, args.output, args.ids, args.meta, args.samples, args.seed)


if __name__ == "__main__":
    main()
