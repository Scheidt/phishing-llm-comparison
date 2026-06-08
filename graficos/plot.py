"""
Orquestrador de graficos.

Escolha quais modelos plotar (MODELOS) e quais graficos gerar (PLOT_*),
e rode `python plot.py` (a partir de qualquer diretorio). Gera, para cada
modelo selecionado, todos os graficos habilitados. As figuras sao salvas
em graficos/images/.

Cada apelido em MODELOS precisa existir em MODELOS_DISPONIVEIS, que mapeia
apelido -> (nome de exibicao, caminho do log). O limiar da matriz de confusao
e calculado automaticamente (menor valor que maximiza o F1).
"""

# ============================================================
# Catalogo de modelos: apelido -> (nome de exibicao, caminho do log)
MODELOS_DISPONIVEIS = {
    "Claude":   ("Claude Haiku 4.5", "results/llm_logs/claude-haiku-4-5-20251001_20260607_004626.csv"),
    "Gemma3":   ("Gemma 3 4B QAT",   "results/llm_logs/gemma3_4b_qat_20260607_015900.csv"),
    "Qwen3":    ("Qwen 3 4B",        "results/llm_logs/qwen3_4b_20260607_065655.csv"),
    "Granite4": ("Granite 4 Tiny",   "results/llm_logs/granite4_tiny_20260607_130903.csv"),
}

# Modelos a plotar (apelidos do catalogo acima)
MODELOS = ["Claude", "Gemma3", "Qwen3", "Granite4"]

# Quais graficos gerar para cada modelo
PLOT_ROC = True       # curva ROC (plot_roc.py)
PLOT_LIMIAR = True   # F1 x limiar (plot_limiar.py)
PLOT_MATRIZ = True    # matriz de confusao (plot_matriz_confusao.py)
# ============================================================

import plot_roc
import plot_limiar
import plot_matriz_confusao


def main():
    for apelido in MODELOS:
        if apelido not in MODELOS_DISPONIVEIS:
            print(f"[aviso] modelo desconhecido, pulando: {apelido}")
            continue

        nome, log_path = MODELOS_DISPONIVEIS[apelido]
        print(f"\n########## {nome} ##########")

        if PLOT_ROC:
            plot_roc.gerar(nome, log_path)
        if PLOT_LIMIAR:
            plot_limiar.gerar(nome, log_path)
        if PLOT_MATRIZ:
            plot_matriz_confusao.gerar(nome, log_path)


if __name__ == "__main__":
    main()
