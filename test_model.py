# run_single.py — valide antes de rodar o benchmark completo
from dataset.dataset import load_dataset
from test_runner import run_model_tests
from metrics import compute_metrics
from models.dmr_client import DmrClient
from models.claude_client import ClaudeClient
from config import LOCAL_MODELS

dataset = load_dataset().head(10)  # apenas 10 amostras

"""
client = DmrClient(model_tag=LOCAL_MODELS["granite4_tiny"])
client.warm_up()

results = run_model_tests("granite4_tiny", client, dataset)
compute_metrics("granite4_tiny", results)
"""


client = ClaudeClient()
results = run_model_tests("claude_haiku", client, dataset)
compute_metrics("claude_haiku", results)
