# run_single.py — valide antes de rodar o benchmark completo
from dataset import load_dataset
from test_runner import run_model_tests
from metrics import compute_metrics
from models.drm_client import DmrClient

dataset = load_dataset().head(10)  # apenas 10 amostras

client = DmrClient(model_tag="ai/gemma3-qat")
client.warm_up()

results = run_model_tests("gemma3_4b_qat", client, dataset)
compute_metrics("gemma3_4b_qat", results)