# test_model.py — smoke test: valide um modelo antes de rodar o benchmark completo.
#
# Roda poucas amostras (N_SAMPLES) em um modelo real e imprime as métricas.
# Serve também para testar a RETOMADA por e-mail (resume) em um modelo real:
#   1. Rode uma vez e interrompa no meio (Ctrl+C ou deixe cair).
#   2. Ponha RESUME = True e rode de novo: deve reaproveitar os e-mails já
#      gravados no log e continuar de onde parou, no MESMO arquivo de log.
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
from metrics import compute_metrics
from models.dmr_client import DmrClient
from models.claude_client import ClaudeClient
from config import LOCAL_MODELS

# ============================================================================
# CONFIGURAÇÃO DO TESTE  (edite à vontade)
# ============================================================================
MODEL_KEY = "granite4_tiny"   # chave em config.LOCAL_MODELS
N_SAMPLES = 10                # quantas amostras do dataset usar
RESUME    = False             # True = retoma de um log parcial existente
# ============================================================================

# Nome do modelo usado nos arquivos de saída/log. Sufixo "_smoketest" para o
# log da retomada deste teste NÃO se misturar com o do benchmark real
# (main.py grava em llm_logs/<MODEL_KEY>_*.csv).
MODEL_NAME = f"{MODEL_KEY}_smoketest"

dataset = load_dataset().head(N_SAMPLES)

client = DmrClient(model_tag=LOCAL_MODELS[MODEL_KEY])
client.warm_up()

results = run_model_tests(MODEL_NAME, client, dataset, resume=RESUME)
compute_metrics(MODEL_NAME, results)


"""
client = ClaudeClient()
results = run_model_tests("claude_haiku", client, dataset, resume=RESUME)
compute_metrics("claude_haiku", results)
"""
