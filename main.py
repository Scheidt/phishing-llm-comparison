"""
Ponto de entrada principal.
Executa o benchmark de forma sequencial: Claude → Gemma3 → Qwen3 → Phi-4 Mini.

O Docker Model Runner faz o swap de modelos automaticamente ao receber
uma requisição para um modelo diferente do que está em memória —
basta garantir que as chamadas sejam sequenciais (não paralelas),
o que este código já faz por natureza.
"""
import sys
from dataset import load_dataset
from test_runner import run_model_tests
from metrics import compute_metrics, save_comparison_report
from models.claude_client import ClaudeClient
from models.dmr_client import DmrClient
from config import LOCAL_MODELS, CLAUDE_MODEL


def main():
    print("\n" + "="*60)
    print("BENCHMARK — DETECÇÃO DE PHISHING COM IA")
    print("="*60)

    dataset = load_dataset()
    all_metrics = []

    # ══════════════════════════════════════════════════════════════
    # BLOCO 1: Claude (API Anthropic)
    # ══════════════════════════════════════════════════════════════
    print("\n[1/4] Claude (API Anthropic)")
    try:
        claude_client = ClaudeClient()
        results_claude = run_model_tests(
            model_name=f"claude_{CLAUDE_MODEL}",
            client=claude_client,
            dataset=dataset,
        )
        metrics_claude = compute_metrics(
            model_name=f"claude_{CLAUDE_MODEL}",
            results=results_claude,
        )
        all_metrics.append(metrics_claude)
    except Exception as e:
        print(f"  ERRO no Claude: {e}")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════
    # BLOCO 2: Modelos Locais via Docker Model Runner — SEQUENCIAL
    # O DMR descarrega o modelo anterior automaticamente ao iniciar
    # o warm_up() do próximo.
    # ══════════════════════════════════════════════════════════════
    model_list = list(LOCAL_MODELS.items())

    for idx, (model_name, model_tag) in enumerate(model_list, start=2):
        print(f"\n[{idx}/{1 + len(model_list)}] {model_name} ({model_tag})")

        client = DmrClient(model_tag=model_tag)

        # Pré-carrega o modelo na memória.
        # Esta chamada também aciona o swap automático do modelo anterior.
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

    # ══════════════════════════════════════════════════════════════
    # RELATÓRIO FINAL
    # ══════════════════════════════════════════════════════════════
    if all_metrics:
        save_comparison_report(all_metrics)

    print("\nBenchmark concluído.")


if __name__ == "__main__":
    main()