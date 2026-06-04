import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"
RUN_CLAUDE        = True

# ── Docker Model Runner ────────────────────────────────────────────────
DMR_BASE_URL = "http://localhost:12434/engines/llama.cpp/v1"
DMR_TIMEOUT  = 120  # segundos

# Mapeamento: nome amigável → tag do modelo no DMR
LOCAL_MODELS = {
    "gemma3_4b_qat": "ai/gemma3-qat:4B-Q4_K_M",
    "qwen3_4b":      "ai/qwen3:4B-UD-Q8_K_XL",
    "granite4_tiny": "ai/granite-4.0-h-tiny:7B-Q4_K_M",
}

# ── Dataset ────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/emails_dataset.example.csv" # TODO: atualizar para o caminho real do dataset

# ── Saídas ─────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
REPORTS_DIR   = f"{RESULTS_DIR}/reports"
LLM_LOGS_DIR = f"{RESULTS_DIR}/llm_logs"
COMPARATIVE_REPORT_FILENAME = f"{REPORTS_DIR}/comparison_report.csv"

# ── Logging ────────────────────────────────────────────────────────────
ENABLE_LLM_LOGGING = True  # Se False, não gera CSVs em llm_logs

# ── Retomada (resume) ──────────────────────────────────────────────────
# Se True, pula modelos que já possuem metrics_<modelo>.json em reports/.
# Serve para se o teste for interrompido, permitindo retomar o teste de onde deu erro.
SKIP_COMPLETED_MODELS = False

# ── Parâmetros de geração ──────────────────────────────────────────────
MAX_TOKENS   = 512   
TEMPERATURE  = 0.0   # Alterar conforme necessário para testar criatividade vs. precisão

# ── Controle de erros ──────────────────────────────────────────────────
MAX_RETRIES  = 3     # Tentativas em caso de falha de rede
RETRY_DELAY  = 2.0   # Segundos entre tentativas