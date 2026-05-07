
# Configurações centralizadas do benchmark.

import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"  # Mais econômico para testes em volume

# ── Docker Model Runner ────────────────────────────────────────────────
DMR_BASE_URL = "http://localhost:12434/engines/llama.cpp/v1"
DMR_API_KEY  = "dockermodelrunner"  # Qualquer string — o DMR não valida a chave
DMR_TIMEOUT  = 120  # segundos

# Mapeamento: nome amigável → tag do modelo no DMR
LOCAL_MODELS = {
    "gemma3_4b_qat": "ai/gemma3-qat:4B-Q4_K_M",
    "qwen3_4b":      "ai/qwen3:4B-UD-Q8_K_XL",
    "granite4_tiny": "ai/granite-4.0-h-tiny:7B-Q4_K_M",
}

# ── Dataset ────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/emails_dataset.csv"

# ── Saídas ─────────────────────────────────────────────────────────────
RESULTS_DIR  = "results"
LOGS_DIR     = f"{RESULTS_DIR}/logs"
REPORTS_DIR  = f"{RESULTS_DIR}/reports"

# ── Parâmetros de geração ──────────────────────────────────────────────
MAX_TOKENS   = 10    # A classificação é apenas uma palavra
TEMPERATURE  = 0.0   # Determinístico — essencial para reprodutibilidade

# ── Controle de erros ──────────────────────────────────────────────────
MAX_RETRIES  = 3     # Tentativas em caso de falha de rede
RETRY_DELAY  = 2.0   # Segundos entre tentativas