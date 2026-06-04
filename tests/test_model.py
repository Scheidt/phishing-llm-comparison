# test_model.py — valide antes de rodar o benchmark completo
import os
import sys

# Permite rodar como script standalone a partir da pasta tests/:
# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.dataset import load_dataset
from model_runner import run_model_tests
from metrics import compute_metrics
from models.dmr_client import DmrClient
from models.claude_client import ClaudeClient
from config import LOCAL_MODELS

dataset = load_dataset().head(10)  # apenas 10 amostras

model_tag = LOCAL_MODELS["granite4_tiny"]

client = DmrClient(model_tag=LOCAL_MODELS[model_tag])
client.warm_up()

results = run_model_tests(model_tag, client, dataset)
compute_metrics(model_tag, results)


"""
client = ClaudeClient()
results = run_model_tests("claude_haiku", client, dataset)
compute_metrics("claude_haiku", results)
"""