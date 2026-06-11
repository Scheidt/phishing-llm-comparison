# test_specific_ids.py — re-teste de e-mails específicos com mais tokens de saída.
#
# Serve para reprocessar os e-mails cujo JSON veio CORTADO (truncado pelo limite
# de tokens) no benchmark. Em vez de rodar o dataset inteiro, você força só os
# IDs problemáticos (FORCED_IDS) em um único modelo (MODEL_KEY), agora com um
# limite de tokens de saída maior (MAX_OUTPUT_TOKENS) para o JSON caber inteiro.
#
# Os IDs cortados por modelo saem do relatório de analysis/error_correction.py
# (seção "RESPOSTAS CORTADAS").
import os
import sys

# Permite rodar como script standalone a partir da pasta tests/:
# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# O console do Windows (cp1252) não imprime alguns caracteres do relatório
# (ex.: '─'); força UTF-8 para o teste rodar em qualquer terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dataset.dataset import load_dataset
from model_runner import run_model_tests
from apply_thresholds import score_run
from metrics import compute_metrics
from models.dmr_client import DmrClient
from config import LOCAL_MODELS

# ============================================================================
# CONFIGURAÇÃO DO TESTE  (edite à vontade)
# ============================================================================
MODEL_KEY = "llama3_3b"   # chave em config.LOCAL_MODELS (o modelo a re-testar)

# IDs dos e-mails a reprocessar (os que vieram com JSON cortado).
# Ex.: gemma3 -> [154, 289, 468, ...] / granite4 -> [591, 928, 1334]
FORCED_IDS = [450, 709, 968, 1163, 1345]

# Limite de tokens da resposta, MAIOR que config.MAX_TOKENS (800), para o JSON
# não ser cortado de novo.
MAX_OUTPUT_TOKENS = 2000

RESUME = False                # True = retoma de um log parcial existente
# ============================================================================

# Nome do modelo usado nos arquivos de saída/log. Sufixo "_retest" para o log
# deste re-teste NÃO sobrescrever o log do benchmark real
# (main.py grava em llm_logs/<MODEL_KEY>_*.csv).
MODEL_NAME = f"{MODEL_KEY}_retest"


class _ClienteTokensMaiores:
    """Embrulha o DmrClient forçando um limite de tokens de saída maior.

    O run_model_tests chama client.classify(system_prompt, user_prompt) sem
    repassar max_tokens; este wrapper injeta MAX_OUTPUT_TOKENS na chamada para
    que as respostas cortadas tenham espaço para vir inteiras.
    """

    def __init__(self, model_tag: str, max_tokens: int):
        self._client = DmrClient(model_tag=model_tag)
        self._max_tokens = max_tokens

    def warm_up(self):
        self._client.warm_up()

    def classify(self, system_prompt: str, user_prompt: str) -> dict:
        return self._client.classify(system_prompt, user_prompt,
                                     max_tokens=self._max_tokens)


# Carrega o dataset e filtra só os IDs pedidos (preservando a ordem de FORCED_IDS).
dataset = load_dataset()
dataset = dataset[dataset["id"].isin(FORCED_IDS)]

encontrados = set(dataset["id"].tolist())
faltando = [i for i in FORCED_IDS if i not in encontrados]
if faltando:
    print(f"[aviso] IDs não encontrados no dataset (ignorados): {faltando}")
if dataset.empty:
    sys.exit("Nenhum dos FORCED_IDS existe no dataset. Verifique os IDs.")

# Reordena para respeitar a ordem de FORCED_IDS.
dataset = dataset.set_index("id").loc[[i for i in FORCED_IDS if i in encontrados]].reset_index()

print(f"Re-testando {len(dataset)} e-mail(s) em '{MODEL_KEY}' "
      f"com max_tokens={MAX_OUTPUT_TOKENS}.")

client = _ClienteTokensMaiores(LOCAL_MODELS[MODEL_KEY], MAX_OUTPUT_TOKENS)
client.warm_up()

results = run_model_tests(MODEL_NAME, client, dataset, resume=RESUME)
# A inferência só grava os fatos; o veredito (predicted_label) é decidido pelo
# scorer. Sem este passo, predicted_label fica em branco e compute_metrics
# quebra. (No pipeline real, main.py chama score_run entre os dois.)
results = score_run(MODEL_NAME, results)
compute_metrics(MODEL_NAME, results)
