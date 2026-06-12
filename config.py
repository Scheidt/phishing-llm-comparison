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
    "llama3_3b":      "ai/llama3.2:3B-Q4_0",
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
# Permite retomar um benchmark interrompido sem repetir os modelos concluídos.
# Caso tenha restado um log parcial, recomenda-se removê-lo antes, para não
# misturar métricas de execuções diferentes.
SKIP_COMPLETED_MODELS = True

# Retomada POR E-MAIL dentro de um mesmo modelo (resume parcial).
# Se True, ao reiniciar um modelo interrompido no meio, reaproveita os e-mails
# já gravados no log parcial (results/llm_logs/<modelo>_<timestamp>.csv) e
# retoma a partir do último e-mail testado, que é re-testado para garantir que
# foi gravado corretamente. A gravação continua no MESMO arquivo de log.
# Requer ENABLE_LLM_LOGGING = True (a retomada lê o log parcial).
# Para forçar um modelo a executar do zero, apague o log dele em llm_logs/.
RESUME_PARTIAL_MODEL = False

# =-=-= Parâmetros de geração =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=
MAX_TOKENS   = 800
TEMPERATURE  = 0.0   # Alterar conforme necessário para testar criatividade vs. precisão

# =-=-= Decisão de rótulo (feita no SCORING, não na inferência) =-=-=-=-=-=-=
# O rótulo final (PHISHING/LEGÍTIMO) e o acerto NÃO são decididos durante a
# inferência. A inferência apenas grava os fatos da resposta (entre eles a nota
# `phishing_likelihood`); o rótulo e o acerto são definidos pelo scorer
# (apply_thresholds.py), chamado após cada modelo. Esse desenho desacopla a
# coleta do julgamento e permite re-decidir tudo apenas alterando os cortes
# abaixo e re-executando o scorer, sem re-inferir nenhum modelo.
#
# O rótulo deriva da nota numérica `phishing_likelihood`, não do campo textual
# `classification`. Em modelos pequenos, a nota mostrou-se mais calibrada que a
# string, que costuma indicar "phishing" mesmo com likelihood baixo (ex.:
# likelihood 10 + classification "phishing"), o que gera falsos positivos.
# Priorizar o número melhorou a precisão de todos os modelos locais sem piorar
# nenhum. O `classification` permanece no schema e no prompt (induz o modelo a
# registrar uma decisão explícita), mas é apenas registrado para análise.
#
# Regra do scorer: phishing_likelihood >= corte DO MODELO => PHISHING; sem nota
# numérica utilizável => erro (predicted_label = -1).
#
# Corte padrão (fallback) para modelos sem corte próprio:
PHISHING_LIKELIHOOD_THRESHOLD = 50

# Cortes (thresholds) POR MODELO. Cada modelo tem um ponto de corte calibrado,
# pois a escala da nota varia de modelo para modelo e um corte único não seria
# o ideal. A chave é o nome do modelo EXATAMENTE como aparece na coluna 'model'
# dos logs (nome amigável nos locais; CLAUDE_MODEL no Claude). Modelos sem corte
# próprio caem na chave "Default". Lido pelo scorer (apply_thresholds.py).
PHISHING_LIKELIHOOD_THRESHOLDS = {
    CLAUDE_MODEL:    76.5,   # claude-haiku-4-5-20251001
    "gemma3_4b_qat": 80.0,
    "qwen3_4b":      92.5,
    "granite4_tiny": 82.5,
    "llama3_3b":     45.0,
    "Default":       PHISHING_LIKELIHOOD_THRESHOLD,  # 50
}

# =-=-= Truncamento do corpo do e-mail =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=
TRUNCATE_EMAIL_BODY = True  # Se True, trunca o corpo do e-mail em EMAIL_BODY_MAX_CHARS
EMAIL_BODY_MAX_CHARS = 5000  # Tamanho máximo do corpo (caracteres) ao truncar

# =-=-= Controle de erros =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=
MAX_RETRIES  = 3     # Tentativas em caso de falha de rede
RETRY_DELAY  = 2.0   # Segundos entre tentativas

# =-=-= Constrained decoding (saída estruturada) =-=-=-=-=-=-=-=-=-=-==-=-=-=
# Se True, força os modelos a responderem exatamente no schema JSON abaixo via
# constrained decoding (grammar/structured outputs), em vez de apenas pedir o
# formato no prompt. O DMR usa response_format (formato OpenAI/json_schema) e o
# Claude usa output_config.format (Anthropic). O prompt continua descrevendo o
# formato, já que a variante "schema + prompt descritivo" é a de melhor
# rendimento.
USE_CONSTRAINED_DECODING = False

# Schema canônico da resposta. Cada cliente o encapsula no formato nativo da
# sua API a partir desta única definição. Observação: o min/max de
# phishing_likelihood não é garantido de forma confiável pelo grammar (e o SDK
# da Anthropic remove a restrição antes do envio); output_parser.py valida o
# intervalo 0..100 de qualquer forma.
RESPONSE_SCHEMA_NAME = "phishing_analysis"
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "indicators": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Observações concretas sobre o email",
        },
        "phishing_likelihood": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "classification": {
            "type": "string",
            "enum": ["phishing", "legitimate"],
        },
    },
    "required": ["indicators", "phishing_likelihood", "classification"],
    "additionalProperties": False,
}