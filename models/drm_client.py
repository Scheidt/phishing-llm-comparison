"""
Cliente para o Docker Model Runner (DMR).

Usa a lib `openai` apontando para o endpoint local do DMR,
que expõe uma API 100% compatível com o formato OpenAI.

Gerenciamento de memória:
  O DMR carrega um modelo por vez automaticamente. Quando uma
  requisição chega para um modelo diferente do que está em memória,
  o DMR faz o swap sozinho — sem necessidade de lógica manual.
  Basta executar os modelos sequencialmente no código Python.
"""
import time
from openai import OpenAI, APIConnectionError, APITimeoutError
from config import (DMR_BASE_URL, DMR_API_KEY, DMR_TIMEOUT,
                    MAX_TOKENS, TEMPERATURE, MAX_RETRIES, RETRY_DELAY)


class DmrClient:
    def __init__(self, model_tag: str):
        """
        model_tag: tag do modelo no DMR
                   Exemplos: 'ai/gemma3-qat'
                             'ai/qwen3:4B-Q4_K_M'
                             'hf.co/bartowski/Phi-4-mini-instruct-GGUF:Q4_K_M'
        """
        self.model_tag = model_tag
        self.client = OpenAI(
            base_url=DMR_BASE_URL,
            api_key=DMR_API_KEY,
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

    def classify(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Envia o prompt para o DMR e retorna resultado com métricas.

        Retorna dict com:
          raw_response, elapsed_seconds, input_tokens, output_tokens, error
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.perf_counter()

                response = self.client.chat.completions.create(
                    model=self.model_tag,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )

                elapsed = time.perf_counter() - start
                raw     = response.choices[0].message.content or ""

                return {
                    "raw_response":    raw,
                    "elapsed_seconds": round(elapsed, 4),
                    "input_tokens":    response.usage.prompt_tokens     if response.usage else 0,
                    "output_tokens":   response.usage.completion_tokens if response.usage else 0,
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