"""
Cliente para o Docker Model Runner (DMR).

Usa a lib `openai` apontando para o endpoint local do DMR,
que expõe uma API compatível com o formato OpenAI.
"""
import time
from openai import OpenAI, APIConnectionError, APITimeoutError
from config import (DMR_BASE_URL, DMR_TIMEOUT, MAX_TOKENS,
                    TEMPERATURE, MAX_RETRIES, RETRY_DELAY)


class DmrClient:
    def __init__(self, model_tag: str):
        """
            Inicializa o cliente do DMR para um modelo específico.
        Args:
            model_tag: tag do modelo no DMR. Verificar modelos existentes em config.py → LOCAL_MODELS.
        """
        self.model_tag = model_tag
        self.client = OpenAI(
            base_url=DMR_BASE_URL,
            api_key="youtube.com/watch?v=nVLZIgcLmsc&pp=ygUdbW9tIGdpdmUgbWUgc29tZSBtb25leSBwbGVhc2U%3D", # O DMR não exige chave, mas a lib OpenAI requer uma string.
            timeout=DMR_TIMEOUT,
        )

    def warm_up(self):
        """
        Envia uma requisição mínima para pré-carregar o modelo na memória
        antes de iniciar os testes. Evita que o primeiro e-mail sofra
        penalidade de tempo pelo carregamento inicial do modelo.
        O DMR automaticamente descarrega o modelo anterior ao receber
        a primeira requisição para este novo modelo.
        """
        print(f"  Pré-carregando '{self.model_tag}'...")
        try:
            self.client.chat.completions.create(
                model=self.model_tag,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            print(f"  Modelo '{self.model_tag}' pronto.")
        except Exception as e:
            print(f"  Aviso: falha no warm-up ({e}). Continuando...")

    def classify(self, system_prompt: str, user_prompt: str,
                 max_tokens: int | None = None) -> dict:
        """
        Envia o prompt para o DMR e retorna resultado com métricas.
        Args:
            system_prompt: Instruções para o modelo
            user_prompt: O conteúdo do email a ser classificado.
            max_tokens: Limite de tokens da resposta. Se None, usa MAX_TOKENS do config.
                        (Útil para testes que exigem respostas maiores, ex: JSON.)
        Returns:
            Retorna dict com:
            raw_response, elapsed_seconds, input_tokens, output_tokens, error
        """
        if max_tokens is None:
            token_limit = MAX_TOKENS
        else:
            token_limit = max_tokens

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.perf_counter()

                response = self.client.chat.completions.create(
                    model=self.model_tag,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=token_limit,
                    temperature=TEMPERATURE,
                )

                elapsed = time.perf_counter() - start
                raw     = response.choices[0].message.content or ""

                # Nem todo backend devolve as contagens de uso de tokens.
                if response.usage is not None:
                    input_tokens  = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                else:
                    input_tokens  = 0
                    output_tokens = 0

                return {
                    "raw_response":    raw,
                    "elapsed_seconds": round(elapsed, 4),
                    "input_tokens":    input_tokens,
                    "output_tokens":   output_tokens,
                    "error":           None,
                }

            except APITimeoutError:
                if attempt == MAX_RETRIES:
                    return {"raw_response": "", "elapsed_seconds": DMR_TIMEOUT,
                            "input_tokens": 0, "output_tokens": 0,
                            "error": f"Timeout após {DMR_TIMEOUT}s"}
                time.sleep(RETRY_DELAY)

            except APIConnectionError as e:
                if attempt == MAX_RETRIES:
                    return {"raw_response": "", "elapsed_seconds": 0,
                            "input_tokens": 0, "output_tokens": 0,
                            "error": (f"DMR indisponível: {e}. "
                                      "Verifique se o Docker Model Runner está ativo.")}
                time.sleep(RETRY_DELAY)

            except Exception as e:
                if attempt == MAX_RETRIES:
                    return {"raw_response": "", "elapsed_seconds": 0,
                            "input_tokens": 0, "output_tokens": 0,
                            "error": str(e)}
                time.sleep(RETRY_DELAY)