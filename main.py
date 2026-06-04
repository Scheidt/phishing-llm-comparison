"""
Ponto de entrada principal.
Executa o benchmark de forma sequencial: Claude → Modelos no DMR conforme config.py.

O Docker Model Runner faz o swap de modelos automaticamente ao receber
uma requisição para um modelo diferente do que está em memória, não rodando
modelos simultaneamente.
"""
import os
import sys
import json
from dataset.dataset import load_dataset
from test_runner import run_model_tests
from metrics import compute_metrics, save_comparison_report
from models.claude_client import ClaudeClient
from models.dmr_client import DmrClient
from config import (LOCAL_MODELS, RUN_CLAUDE, CLAUDE_MODEL,
                    REPORTS_DIR, SKIP_COMPLETED_MODELS)


def _load_existing_metrics(model_name: str) -> dict | None:
    """
    Retorna as métricas já salvas de um modelo (reports/metrics_<modelo>.json),
    ou None se ainda não existirem. Usado para retomar benchmarks interrompidos.
    """
    path = os.path.join(REPORTS_DIR, f"metrics_{model_name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    print(f"\n{'=-'*30 + '='}")
    print("\n"*2)

    dataset = load_dataset()
    all_metrics = []

    # Claude (API Anthropic)
    if RUN_CLAUDE:
        print(f"[1/{1 + len(LOCAL_MODELS)}] Claude (API Anthropic)")
        existing = _load_existing_metrics(CLAUDE_MODEL) if SKIP_COMPLETED_MODELS else None
        if existing is not None:
            print(f"  Pulando {CLAUDE_MODEL}: metrics já existe em reports/.")
            all_metrics.append(existing)
        else:
            try:
                claude_client = ClaudeClient()
                results_claude = run_model_tests(
                    model_name=CLAUDE_MODEL,
                    client=claude_client,
                    dataset=dataset,
                )
                metrics_claude = compute_metrics(
                    model_name=CLAUDE_MODEL,
                    results=results_claude,
                )
                all_metrics.append(metrics_claude)
            except Exception as e:
                print(f"  ERRO no Claude: {e}")
                sys.exit(1)

    # Modelos Locais via Docker Model Runner
    # O DMR descarrega o modelo anterior automaticamente ao iniciar
    # o warm_up() do próximo.
    model_list = list(LOCAL_MODELS.items())

    for idx, (model_name, model_tag) in enumerate(model_list, start=2):
        print(f"\n{'=-'*30 + '='}")
        print(f"\n[{idx}/{1 + len(model_list)}] {model_name} ({model_tag})")

        if SKIP_COMPLETED_MODELS:
            existing = _load_existing_metrics(model_name)
            if existing is not None:
                print(f"  Pulando {model_name}: metrics já existe em reports/.")
                all_metrics.append(existing)
                continue

        client = DmrClient(model_tag=model_tag)

        # Pré-carrega o modelo na memória.
        client.warm_up()

        try:
            results = run_model_tests(
                model_name=model_name,
                client=client,
                dataset=dataset,
            )
            metrics = compute_metrics(
                model_name=model_name,
                results=results,
            )
            all_metrics.append(metrics)

        except Exception as e:
            print(f"  ERRO em {model_name}: {e}")


    # RELATÓRIO FINAL
    if all_metrics:
        save_comparison_report(all_metrics)

    print("\nBenchmark concluído.")


if __name__ == "__main__":
    main()