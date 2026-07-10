"""
Executa o benchmark.
"""
import sys
import apply_thresholds
from dataset.dataset import load_dataset
from model_runner import run_model_tests
from metrics import compute_metrics, save_comparison_report, load_existing_metrics
from models.claude_client import ClaudeClient
from models.dmr_client import DmrClient
from config import (LOCAL_MODELS, RUN_CLAUDE, CLAUDE_MODEL,
                    SKIP_COMPLETED_MODELS, RESUME_PARTIAL_MODEL)


def main():
    print(f"\n{'=-'*30 + '='}")
    print("\n"*2)

    dataset = load_dataset().head(3)
    all_metrics = []

    # Claude
    if RUN_CLAUDE:
        print(f"[1/{1 + len(LOCAL_MODELS)}] Claude")
        existing = None
        if SKIP_COMPLETED_MODELS:
            existing = load_existing_metrics(CLAUDE_MODEL)
        if existing is not None:
            print(f"  Pulando {CLAUDE_MODEL}: já há métricas em reports/.")
            all_metrics.append(existing)
        else:
            try:
                claude_client = ClaudeClient()
                results_claude = run_model_tests(
                    model_name=CLAUDE_MODEL,
                    client=claude_client,
                    dataset=dataset,
                    resume=RESUME_PARTIAL_MODEL,
                )
                # Scoring desacoplado: a inferência só coletou os fatos;
                # aplica-se agora o corte do modelo e calcula-se o acerto.
                results_claude = apply_thresholds.score_run(CLAUDE_MODEL, results_claude)
                metrics_claude = compute_metrics(
                    model_name=CLAUDE_MODEL,
                    results=results_claude,
                )
                all_metrics.append(metrics_claude)
            except Exception as e:
                print(f"  ERRO no Claude: {e}")
                sys.exit(1)

    # Modelos Locais via Docker Model Runner
    model_list = list(LOCAL_MODELS.items())

    for idx, (model_name, model_tag) in enumerate(model_list, start=2):
        print(f"\n{'=-'*30 + '='}")
        print(f"\n[{idx}/{1 + len(model_list)}] {model_name} ({model_tag})")

        if SKIP_COMPLETED_MODELS:
            existing = load_existing_metrics(model_name)
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
                resume=RESUME_PARTIAL_MODEL,
            )
            # Scoring desacoplado: aplica o corte do modelo e calcula o acerto.
            results = apply_thresholds.score_run(model_name, results)
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