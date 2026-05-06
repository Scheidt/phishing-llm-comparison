"""
Carregamento e validação do dataset de e-mails.
"""
import pandas as pd
from config import DATASET_PATH


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Carrega o dataset CSV e valida as colunas obrigatórias.
    Retorna um DataFrame com colunas: id, subject, body, label
    """
    df = pd.read_csv(path)

    required_cols = {"id", "subject", "body", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset está faltando colunas: {missing}")

    df["id"]      = df["id"].astype(int)
    df["label"]   = df["label"].astype(int)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"]    = df["body"].fillna("").astype(str)

    invalid = df[~df["label"].isin([0, 1])]
    if not invalid.empty:
        raise ValueError(f"Labels inválidos encontrados: {invalid['label'].unique()}")

    print(f"Dataset carregado: {len(df)} amostras "
          f"({df['label'].sum()} phishing, {(df['label']==0).sum()} legítimos)")
    return df