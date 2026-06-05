"""
Cliente para a API da Anthropic (Claude).
"""
import time
import anthropic
from config import (ANTHROPIC_API_KEY, CLAUDE_MODEL,
                    MAX_TOKENS, TEMPERATURE, MAX_RETRIES, RETRY_DELAY,
                    USE_CONSTRAINED_DECODING, RESPONSE_SCHEMA)


# Constrained decoding no Claude: a API da Anthropic não aceita o response_format
# do OpenAI; o equivalente é output_config.format com um json_schema (structured
# outputs, suportado pelo Haiku 4.5). O mesmo RESPONSE_SCHEMA do config é usado.
# As restrições min/max não suportadas são removidas pelo SDK antes do envio.
_OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}


class ClaudeClient:
    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY não encontrada no .env")
        self.client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model_name = CLAUDE_MODEL

    def classify(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Envia prompts para o Claude e retorna resultado com métricas de tempo.
        Args:
            system_prompt: Instruções para o modelo
            user_prompt: O conteúdo do email a ser classificado.
        Returns:
            Retorna dict com:
            raw_response, elapsed_seconds, input_tokens, output_tokens, error
        """
        # Aplica constrained decoding apenas se habilitado no config.
        extra = {"output_config": _OUTPUT_CONFIG} if USE_CONSTRAINED_DECODING else {}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.perf_counter()

                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    **extra,
                )

                elapsed = time.perf_counter() - start

                return {
                    "raw_response":    message.content[0].text,
                    "elapsed_seconds": round(elapsed, 4),
                    "input_tokens":    message.usage.input_tokens,
                    "output_tokens":   message.usage.output_tokens,
                    "error":           None,
                }

            except anthropic.RateLimitError:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"  Rate limit. Aguardando {wait}s (tentativa {attempt}/{MAX_RETRIES})")
                time.sleep(wait)

            except anthropic.APIError as e:
                if attempt == MAX_RETRIES:
                    return {"raw_response": "", "elapsed_seconds": 0,
                            "input_tokens": 0, "output_tokens": 0,
                            "error": str(e)}
                time.sleep(RETRY_DELAY)

        return {"raw_response": "", "elapsed_seconds": 0,
                "input_tokens": 0, "output_tokens": 0,
                "error": "Max retries excedido"}