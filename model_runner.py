"""
Executor de testes: itera sobre o dataset e coleta predições de um modelo.
"""
from tqdm import tqdm
import pandas as pd

from prompt import build_prompt
from output_parser import parse_response
from logger import BenchmarkLogger, load_partial_results


def run_model_tests(
    model_name: str,
    client, # ClaudeClient ou DmrClient
    dataset: pd.DataFrame,
    resume: bool = False,
) -> list[dict]:
    """
    Executa o benchmark completo para um modelo.
    Retorna lista de dicionários com os resultados de cada predição.
    Args:
        model_name: Nome do modelo (ex: "claude_haiku", "qwen3_4b")
        client: Instância do cliente (ClaudeClient ou DmrClient)
        dataset: DataFrame com colunas id, from, subject, body, label
        resume: Se True, retoma de um log parcial existente, reaproveitando os
                e-mails já testados (menos o último, que é re-testado) e grava
                no mesmo CSV. Sem log parcial, roda normalmente do zero.
    Returns:
        Lista de registros (fatos da inferência) por e-mail. is_error/was_repaired
        já vêm preenchidos (validade não depende do corte); já as colunas que
        dependem do corte (predicted_label, predicted_label_text, is_correct) vêm
        VAZIAS e devem ser preenchidas pelo scorer (apply_thresholds.py) antes do
        cálculo de métricas.
    """
    print(f"\n{'=-'*30 + '='}")
    print(f"Iniciando testes: {model_name.upper()}")
    print(f"Total de amostras: {len(dataset)}")
    print(f"\n{'=-'*30 + '='}")

    resume_path, results, skip_ids = (
        load_partial_results(model_name) if resume else (None, [], set())
    )

    logger = BenchmarkLogger(model_name, resume_path=resume_path)

    pending = dataset[~dataset["id"].astype(int).isin(skip_ids)]
    if skip_ids:
        print(f"Pulando {len(skip_ids)} e-mail(s) já testado(s); "
              f"restam {len(pending)}.")

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc=model_name):
        email_from = str(row.get("from", "") or "")
        system_prompt, user_prompt = build_prompt(
            subject=row["subject"],
            body=row["body"],
            sender=email_from,
        )

        response = client.classify(system_prompt, user_prompt)

        parsed = (
            parse_response(response["raw_response"])
            if not response["error"]
            else {"classification_text": None,
                  "phishing_likelihood": None, "indicators": [],
                  "parse_note": "", "repaired": False}
        )

        record = logger.log(
            email_id            = int(row["id"]),
            email_from          = email_from,
            true_label          = int(row["label"]),
            raw_response        = response["raw_response"],
            classification_text = parsed["classification_text"],
            phishing_likelihood = parsed["phishing_likelihood"],
            indicators          = parsed["indicators"],
            parse_note          = parsed["parse_note"],
            repaired            = parsed["repaired"],
            elapsed             = response["elapsed_seconds"],
            input_tokens        = response["input_tokens"],
            output_tokens       = response["output_tokens"],
            error               = response["error"],
        )
        results.append(record)

    # Resumo só com fatos da inferência — o acerto é calculado depois, no scorer.
    total         = len(results)
    client_errors = sum(1 for r in results if r["error_message"])
    repaired      = sum(1 for r in results if r["was_repaired"])
    print(f"\n[{model_name}] Inferência concluída: {total} e-mail(s), "
          f"{client_errors} erro(s) de chamada, {repaired} com JSON reparado. "
          f"O acerto é calculado no scoring (apply_thresholds.py).")

    return results