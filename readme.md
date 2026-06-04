# phishing-llm-comparison

Benchmark que compara LLMs na tarefa de classificação de e-mails de phishing.
Compara o Claude (API da Anthropic) contra modelos locais rodando
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
  "reasons": ["motivo curto", "motivo curto"]
}
```

O parser ([output_parser.py](output_parser.py)) extrai esse JSON (tolerando cercas
markdown e texto ao redor) e mapeia `classification` para o rótulo interno:
`1 = PHISHING`, `0 = LEGÍTIMO`. Respostas inválidas ou com falha recebem
`predicted_label = -1` e são contabilizadas como erro.

## Pré-requisitos

- **Python 3.10+** (o código usa anotações de tipo modernas, ex. `dict | None`).
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
| `SKIP_COMPLETED_MODELS` | Se `True`, pula modelos que já têm `metrics_<modelo>.json` (permite retomar um benchmark interrompido). |
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

## Uso

Benchmark completo (Claude + todos os modelos de `LOCAL_MODELS`):

```bash
python main.py
```

Smoke test rápido (10 amostras — valide antes do run completo):

```bash
python test_model.py
```

Teste de consistência de formato JSON de um modelo local (roda o mesmo prompt N
vezes; útil para testar modelos pequenos que quebram o formato em prompts grandes; lê o
e-mail de `test_email.txt`):

```bash
python test_output.py
```

Remontar o relatório comparativo juntando os `metrics_*.json` já existentes (sem
re-executar os modelos):

```bash
python metrics.py --rebuild
```

> Os arquivos `test_*.py` são **scripts executáveis standalone**, não testes
> pytest. Não há framework de testes nem linter configurados.

## Saídas

Tudo é gravado em `results/` (gitignored):

- `results/reports/metrics_<modelo>.json` — métricas de cada modelo.
- `results/reports/comparison_report.csv` — comparativo final, ordenado por F1.
- `results/llm_logs/<modelo>_<timestamp>.csv` — dados de cada e-mail, contendo todas
  informações relacionadas a cada entrada (se `ENABLE_LLM_LOGGING = True`).

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
| [test_runner.py](test_runner.py) | `run_model_tests()` itera o dataset e coleta predições. |
| [logger.py](logger.py) | `BenchmarkLogger` grava um CSV por modelo. |
| [metrics.py](metrics.py) | Calcula métricas (sklearn) e gera o relatório comparativo. |

Os dois clientes são intercambiáveis: ambos expõem
`classify(system_prompt, user_prompt) -> dict` com as chaves `raw_response`,
`elapsed_seconds`, `input_tokens`, `output_tokens` e `error`.

## Tratamento de erros

- Os clientes tratam erros internamente e retornam `error` no dict (não levantam
  exceção por e-mail). Um modelo local que falha **não** derruba o benchmark —
  `main.py` segue para o próximo. Já uma falha no Claude faz `sys.exit(1)`.
- Com `SKIP_COMPLETED_MODELS = True`, é possível retomar um benchmark interrompido,
  pulando modelos que já possuem `metrics_<modelo>.json`.

---

`legacy/` contém scripts antigos (JS/Python) fora do pipeline atual.
