# phishing-llm-comparison

Benchmark que compara LLMs na tarefa de classificação de e-mails de phishing.
Compara o Claude contra modelos locais rodando
no Docker Model Runner, usando exatamente o mesmo prompt para todos os
modelos, e gera um relatório comparativo de métricas (acurácia, precisão, recall,
F1, tempo médio de resposta e taxa de erro).

## Como funciona

Todos os modelos recebem o mesmo prompt (definido em [prompt.py](prompt.py)) e
devem responder com um único objeto JSON:

```json
{
  "classification": "phishing" | "legitimate",
  "phishing_likelihood": 0,
  "indicators": ["motivo curto", "motivo curto"]
}
```

O parser ([output_parser.py](output_parser.py)) extrai esse JSON (tolerando cercas
markdown e texto ao redor) e mapeia `classification` para o rótulo interno:
`1 = PHISHING`, `0 = LEGÍTIMO`. Respostas inválidas ou com falha recebem
`predicted_label = -1` e são contabilizadas como erro.

### Reparo heurístico de JSON

Modelos menores às vezes entregam um JSON quase-válido: cortado no meio (limite de
tokens) ou com aspas duplas literais não escapadas dentro de uma string
(ex.: `"... asks, "what is this?" ..."`). O parser tenta **reparar** esses casos
em dois últimos fallbacks — fechar o que ficou aberto e escapar aspas órfãs.

Quando uma resposta só pôde ser lida **após reparo**, a saída bruta do modelo
**não** era um JSON válido. Por isso ela é marcada com `was_repaired = True` e
**conta como erro** (`is_error = True`) — mas a predição resgatada é
**preservada** em `predicted_label` (não vira `-1`). Isso permite medir as duas
coisas separadamente (ver [Métricas](#métricas)). Localizar/extrair um JSON que já
era bem-formado (cercas markdown, texto ao redor) **não** é considerado reparo.

> O reparo de aspas é uma heurística e roda só como último recurso: uma aspa
> literal seguida logo de `:` (raro) poderia ser confundida com o fim de uma
> chave. No pior caso o `json.loads` rejeita e a resposta cai como erro
> irrecuperável — não há parse silenciosamente errado.

### Métricas

Cada modelo recebe **duas faixas** de métricas, lado a lado no relatório:

- **Bruta (estrito)** — avalia a saída como o modelo a entregou: respostas
  reparadas contam como erro e ficam de fora da acurácia/precisão/recall/F1.
  Mede a qualidade real da formatação do modelo. Chaves: `error_rate`,
  `accuracy`, `f1_score`, `repaired_count`, etc.
- **Pós-reparo (resgatado)** — aproveita as predições resgatadas pelo reparo;
  só sobra como erro o que foi irrecuperável (`predicted_label = -1`). Chaves com
  prefixo `rescued_` (`rescued_error_rate`, `rescued_accuracy`, ...).

A leitura é: *"houve taxa de erro X na saída bruta; ao reparar esses erros com
heurísticas externas, os resultados passam a ser Y"*. A classe positiva é
**PHISHING (1)** e o relatório é ordenado por `f1_score`.

## Pré-requisitos

- **Python 3.10+**.
- **Chave da API da Anthropic** para rodar o Claude.
- **Docker Model Runner** ativo em `http://localhost:12434` para rodar os modelos
  locais. O DMR mantém um modelo por vez em memória e faz o swap automático ao
  receber requisição para outro modelo, por isso o benchmark roda os modelos
  sequencialmente e chama `warm_up()` para pré-carregar cada um (caso contrário, 
  os modelos locais sofreriam uma penalidade de latência do tempo para carregar o
  modelo).

## Instalação

```bash
pip install -r requirements.txt
```

Copie o arquivo de exemplo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Depois abra o `.env` e substitua o valor pela sua chave da API:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Coloque seu dataset em `dataset/.` e configure corretamente o caminho em `config.py`

## Configuração

Toda a configuração fica em [config.py](config.py). Principais pontos:

| Opção | Descrição |
| --- | --- |
| `CLAUDE_MODEL` | Modelo da Anthropic a ser testado. |
| `RUN_CLAUDE` | Liga/desliga o teste do Claude. |
| `LOCAL_MODELS` | Mapeia *nome amigável: tag do modelo no DMR*. |
| `DMR_BASE_URL` | Endpoint do Docker Model Runner. |
| `DATASET_PATH` | Caminho do CSV de e-mails. **Deve ser modificado para o dataset real** |
| `ENABLE_LLM_LOGGING` | Se `True`, grava um CSV por modelo em `results/llm_logs/`. |
| `SKIP_COMPLETED_MODELS` | Se `True`, pula modelos que já têm `metrics_<modelo>.json` (retomada **por modelo** — pula modelos já concluídos). |
| `RESUME_PARTIAL_MODEL` | Se `True`, retoma **por e-mail** um modelo interrompido no meio: reaproveita os e-mails já gravados no log parcial (menos o último, que é re-testado) e continua no mesmo CSV. Requer `ENABLE_LLM_LOGGING = True`. |
| `MAX_TOKENS` / `TEMPERATURE` | Parâmetros de geração compartilhados por todos os modelos. |
| `TRUNCATE_EMAIL_BODY` / `EMAIL_BODY_MAX_CHARS` | Trunca o corpo do e-mail (padrão: 2000 chars). |
| `MAX_RETRIES` / `RETRY_DELAY` | Política de retentativa em caso de falha de rede. |

### Dataset

O CSV precisa ter as colunas obrigatórias:

| Coluna | Descrição |
| --- | --- |
| `id` | Identificador do e-mail. |
| `subject` | Assunto. |
| `body` | Corpo do e-mail. |
| `label` | `0` = legítimo, `1` = phishing. |

#### Reconstrução do dataset CEAS_08

O dataset usado no benchmark é uma amostra balanceada 50/50 (phishing/legítimo)
extraída do dataset **CEAS_08**. O CSV selecionado é grande e **não é versionado** —
o repositório guarda apenas a "receita" para reconstruí-lo de forma determinística:

| Arquivo | Versionado? | Papel |
| --- | --- | --- |
| `dataset/CEAS_08.csv` | Não (baixar) | CSV bruto de origem. |
| `dataset/ceas08_sample_ids.csv` | Sim | Ids das amostras sorteadas. |
| `dataset/ceas08_sample_ids.meta.json` | Sim | Hash SHA-256 do source, semente e nº de amostras. |
| `dataset/emails_dataset.csv` | Não (derivado) | Dataset final, gerado pelos passos abaixo. |

Passos:

1. Baixe o CSV bruto do CEAS_08 em [[LINK](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)] e salve em `dataset/CEAS_08.csv`.

2. **Reconstrua** o dataset a partir dos ids salvos (caminho recomendado — reproduz
   exatamente a amostra original):

   ```bash
   python dataset/prepare_ceas08.py --rebuild
   ```

   O script valida o hash SHA-256 do `CEAS_08.csv` contra o registrado em
   `ceas08_sample_ids.meta.json` e **aborta** se o source for diferente do usado na
   amostragem, garantindo reprodutibilidade.

3. Aponte `DATASET_PATH` em [config.py](config.py) para `dataset/emails_dataset.csv`.

Para gerar uma **nova** amostra (ex.: outro tamanho ou semente) em vez de reconstruir
a existente — isso sobrescreve os ids e metadados salvos:

```bash
python dataset/prepare_ceas08.py --samples 2000 --seed 42
```

## Uso

Benchmark completo (Claude + todos os modelos de `LOCAL_MODELS`):

```bash
python main.py
```

## Para adicionar novos modelos

1. Baixe o modelo no Docker Model Runner (ele precisa estar disponível no DMR).
2. Registre-o em `LOCAL_MODELS` no [config.py](config.py), mapeando um *nome
   amigável* para a *tag do modelo no DMR*:

   ```python
   LOCAL_MODELS = {
       "gemma3_4b_qat": "ai/gemma3-qat:4B-Q4_K_M",
       "meu_modelo":    "ai/novo-modelo:tag",   # novo modelo
   }
   ```

   O *nome amigável* é usado nos arquivos de saída (`metrics_<nome>.json`, logs).

Com isso, `python main.py` já inclui o novo modelo no benchmark. Antes de um benchmark
completo recomenda-se realizar os passos a seguir:

### Passos opcionais (recomendados antes de um benchmark completo)

- **Smoke test** — confirma que o modelo responde e o formato é válido.
  Vá em [test_model.py](tests/test_model.py), ajuste a constante para o nome do
  seu modelo (definido no passo 2) e rode:

  ```bash
  python tests/test_model.py
  ```

- **Consistência de formato JSON** — útil para modelos pequenos, que tendem a
  quebrar o formato em prompts grandes. Roda o mesmo prompt N vezes (lê o e-mail
  de `tests/long_email.txt`):

  ```bash
  python tests/test_output.py
  ```

- **Parser / reparo de JSON** — teste offline (não usa DMR nem a API) que cobre o
  parsing e os fallbacks de reparo, incluindo aspas internas não escapadas e JSON
  truncado:

  ```bash
  python tests/test_parser.py
  ```

Para remontar o relatório comparativo juntando os `metrics_*.json` já existentes
(sem executar nenhum modelo):

```bash
python metrics.py --rebuild
```

> Os scripts em `tests/` (`test_model.py`, `test_output.py`, `test_parser.py`,
> `test_resume.py`) são **scripts executáveis standalone**, não testes pytest,
> rode-os a partir da raiz do projeto. Não há framework de testes nem linter
> configurados.

## Saídas

Tudo é gravado em `results/` (gitignored):

- `results/reports/metrics_<modelo>.json` — métricas de cada modelo.
- `results/reports/comparison_report.csv` — comparativo final, ordenado por F1.
- `results/llm_logs/<modelo>_<timestamp>.csv` — dados de cada e-mail, contendo todas
  informações relacionadas a cada entrada (se `ENABLE_LLM_LOGGING = True`). Inclui a
  coluna `was_repaired` (`True` quando a resposta só pôde ser lida após reparo
  heurístico — conta como erro na faixa bruta, mas a predição é preservada).

## Fluxo / arquitetura

`main.py` orquestra → `load_dataset()` → para cada modelo: `run_model_tests()` →
`compute_metrics()` → `save_comparison_report()`.

| Arquivo | Responsabilidade |
| --- | --- |
| [main.py](main.py) | Orquestra o benchmark; roda os modelos sequencialmente. |
| [config.py](config.py) | Fonte única de configuração. |
| [dataset/dataset.py](dataset/dataset.py) | `load_dataset()` carrega e valida o CSV. |
| [prompt.py](prompt.py) | Prompt compartilhado por todos os modelos; `build_prompt()` trunca o corpo. |
| [output_parser.py](output_parser.py) | `parse_response()` extrai o JSON e mapeia para o rótulo interno. |
| [models/claude_client.py](models/claude_client.py) | `ClaudeClient` — cliente da API Claude. |
| [models/dmr_client.py](models/dmr_client.py) | `DmrClient` — usa a lib `openai` apontada para o endpoint local do DMR; tem `warm_up()`. |
| [model_runner.py](model_runner.py) | `run_model_tests()` itera o dataset e coleta predições. |
| [logger.py](logger.py) | `BenchmarkLogger` grava um CSV por modelo. |
| [metrics.py](metrics.py) | Calcula métricas (sklearn) e gera o relatório comparativo. |

Os dois clientes são intercambiáveis: ambos expõem
`classify(system_prompt, user_prompt) -> dict` com as chaves `raw_response`,
`elapsed_seconds`, `input_tokens`, `output_tokens` e `error`.

## Tratamento de erros

- Os clientes tratam erros internamente e retornam `error` no dict (não levantam
  exceção por e-mail). Um modelo local que falha **não** derruba o benchmark —
  `main.py` segue para o próximo. Já uma falha no Claude faz `sys.exit(1)`.
## Retomada de benchmarks interrompidos

Como cada e-mail é gravado no CSV de log na hora (append imediato, não no fim da
execução), um benchmark que cai no meio pode ser retomado sem refazer tudo. Há dois
níveis, que funcionam em conjunto:

- **Por modelo** — `SKIP_COMPLETED_MODELS = True`: pula modelos que já têm
  `metrics_<modelo>.json` (esse JSON só é escrito quando o modelo termina 100%).
- **Por e-mail** — `RESUME_PARTIAL_MODEL = True`: para o modelo que foi interrompido
  no meio (sem `metrics_*.json`, mas com log parcial), reaproveita os e-mails já
  gravados em `results/llm_logs/<modelo>_<timestamp>.csv`, **re-testa apenas o último**
  (garantindo que a última linha não ficou gravada pela metade) e continua gravando
  no mesmo arquivo de log. As métricas finais cobrem o conjunto completo.

  Requer `ENABLE_LLM_LOGGING = True`. Para forçar um modelo a rodar do zero, apague
  o CSV dele em `results/llm_logs/`.

Basta reexecutar `python main.py`: o programa detecta o estado e retoma de onde parou.

---

