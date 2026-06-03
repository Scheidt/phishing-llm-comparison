"""
Executor de testes: itera sobre o dataset e coleta predições de um modelo.
"""
from tqdm import tqdm
import pandas as pd

from prompt import build_prompt, parse_response
from logger import BenchmarkLogger


def run_model_tests(
    model_name: str,
    client, # ClaudeClient ou DmrClient
    dataset: pd.DataFrame,
) -> list[dict]:
    """
    Executa o benchmark completo para um modelo.
    Retorna lista de dicionários com os resultados de cada predição.
    Args:
        model_name: Nome do modelo (ex: "claude_haiku", "qwen3_4b")
        client: Instância do cliente (ClaudeClient ou DmrClient)
        dataset: DataFrame com colunas id, subject, body, label
    Returns:
        Lista de dicionários para cada email com:
        email_id, true_label, raw_response, predicted, elapsed,
        input_tokens, output_tokens, is_correct, is_error, error
    """
    print(f"\n{'=-'*30 + '='}")
    print(f"Iniciando testes: {model_name.upper()}")
    print(f"Total de amostras: {len(dataset)}")
    print(f"\n{'=-'*30 + '='}")

    logger  = BenchmarkLogger(model_name)
    results = []

    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=model_name):
        system_prompt, user_prompt = build_prompt(
            subject=row["subject"],
            body=row["body"],
        )

        response = client.classify(system_prompt, user_prompt)

        parsed = (
            parse_response(response["raw_response"])
            if not response["error"]
            else {"predicted": None, "phishing_likelihood": None,
                  "reasons": [], "parse_note": ""}
        )

        record = logger.log(
            email_id            = int(row["id"]),
            true_label          = int(row["label"]),
            raw_response        = response["raw_response"],
            predicted           = parsed["predicted"],
            phishing_likelihood = parsed["phishing_likelihood"],
            reasons             = parsed["reasons"],
            parse_note          = parsed["parse_note"],
            elapsed             = response["elapsed_seconds"],
            input_tokens        = response["input_tokens"],
            output_tokens       = response["output_tokens"],
            error               = response["error"],
        )
        results.append(record)

    total   = len(results)
    errors  = sum(1 for r in results if r["is_error"])
    correct = sum(1 for r in results if r["is_correct"])
    print(f"\n[{model_name}] Concluído: {correct}/{total - errors} corretos, {errors} erros")

    return results