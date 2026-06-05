import os
from dotenv import load_dotenv

load_dotenv()

# =-=-= Claude =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"
RUN_CLAUDE        = True

# =-=-= Docker Model Runner =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=
DMR_BASE_URL = "http://localhost:12434/engines/llama.cpp/v1"
DMR_TIMEOUT  = 120  # segundos

# Mapeamento: nome amigável: tag do modelo no DMR
LOCAL_MODELS = {
    "gemma3_4b_qat": "ai/gemma3-qat:4B-Q4_K_M",
    "qwen3_4b":      "ai/qwen3:4B-UD-Q8_K_XL",
    "granite4_tiny": "ai/granite-4.0-h-tiny:7B-Q4_K_M",
}

# =-=-= Dataset =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-=
DATASET_PATH = "dataset/emails_dataset.csv"

# =-=-= Saídas =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-==
RESULTS_DIR   = "results"
REPORTS_DIR   = f"{RESULTS_DIR}/reports"
LLM_LOGS_DIR = f"{RESULTS_DIR}/llm_logs"
COMPARATIVE_REPORT_FILENAME = f"{REPORTS_DIR}/comparison_report.csv"

# =-=-= Logging =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-==-=-=-
ENABLE_LLM_LOGGING = True  # Se False, não gera CSVs em llm_logs

# =-=-= Retomada (resume) =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=
# Se True, pula modelos que já possuem metrics_<modelo>.json em reports/.
# Serve para se o teste for interrompido, permitindo retomar o teste de onde deu erro.
# Se gerou um log parcial, é bom deletar o log anterior, para não misturar métricas de execuções diferentes.
SKIP_COMPLETED_MODELS = False

# Retomada POR E-MAIL dentro de um mesmo modelo (resume parcial).
# Se True, ao reiniciar um modelo interrompido no meio, reaproveita os e-mails
# já gravados no log parcial (results/llm_logs/<modelo>_<timestamp>.csv) e
# retoma a partir do último e-mail testado — que é re-testado, por garantia de
# que foi gravado corretamente. Continua gravando no MESMO arquivo de log.
# Requer ENABLE_LLM_LOGGING = True (a retomada lê o log parcial).
# Para forçar um modelo a rodar do zero, apague o log dele em llm_logs/.
RESUME_PARTIAL_MODEL = True

# =-=-= Parâmetros de geração =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=
MAX_TOKENS   = 512
TEMPERATURE  = 0.0   # Alterar conforme necessário para testar criatividade vs. precisão

# =-=-= Truncamento do corpo do e-mail =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=
TRUNCATE_EMAIL_BODY = True  # Se True, trunca o corpo do e-mail em EMAIL_BODY_MAX_CHARS
EMAIL_BODY_MAX_CHARS = 4000  # Tamanho máximo do corpo (caracteres) ao truncar

# =-=-= Controle de erros =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=
MAX_RETRIES  = 3     # Tentativas em caso de falha de rede
RETRY_DELAY  = 2.0   # Segundos entre tentativas