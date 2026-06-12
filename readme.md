# phishing-llm-comparison

Benchmark que compara modelos de linguagem na classificação binária de e-mails
(phishing ou legítimo). O experimento confronta o Claude (API da Anthropic) com
modelos locais executados no Docker Model Runner (DMR), aplicando o mesmo prompt
a todos os modelos, e gera um relatório comparativo de métricas (acurácia,
precisão, recall, F1-Score, tempo médio de resposta e taxa de erro).

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

O parser ([output_parser.py](output_parser.py)) extrai os campos estruturados
desse JSON, tolerando cercas markdown e texto ao redor. A decisão do rótulo é
feita em etapa posterior pelo scorer ([apply_thresholds.py](apply_thresholds.py)),
que aplica o ponto de corte de cada modelo sobre a nota `phishing_likelihood`:
atribui-se `1 = PHISHING` quando a nota atinge o corte e `0 = LEGÍTIMO` caso
contrário. Respostas inválidas ou com falha recebem `predicted_label = -1` e são
contabilizadas como erro.

### Reparo heurístico de JSON

Modelos menores por vezes produzem um JSON quase válido: truncado pelo limite de
tokens ou com aspas duplas literais não escapadas dentro de uma string
(ex.: `"... asks, "what is this?" ..."`). O parser tenta reparar esses casos em
dois fallbacks finais: o fechamento do que ficou aberto e o escape das aspas
internas.

Quando uma resposta só pôde ser lida após reparo, a saída bruta do modelo não
era um JSON válido. Por isso ela é marcada com `was_repaired = True` e
contabilizada como erro (`is_error = True`), embora a predição resgatada seja
preservada em `predicted_label` (não recebe `-1`). Dessa forma, os dois aspectos
são medidos separadamente (ver [Métricas](#métricas)). Localizar e extrair um
JSON que já era bem-formado (cercas markdown, texto ao redor) não é considerado
reparo.

> O reparo de aspas é uma heurística e roda apenas como último recurso: uma aspa
> literal seguida de `:` (caso raro) poderia ser confundida com o fim de uma
> chave. No pior caso, o `json.loads` rejeita o texto e a resposta é registrada
> como erro irrecuperável, sem risco de um parse silenciosamente incorreto.

### Métricas

Calculam-se duas faixas de métricas por modelo, apresentadas lado a lado no
relatório:

- **Bruta (estrito)**: avalia a saída como o modelo a entregou. Respostas
  reparadas contam como erro e ficam fora da acurácia, precisão, recall e
  F1-Score. Essa faixa mede a qualidade de formatação do modelo. Chaves:
  `error_rate`, `accuracy`, `f1_score`, `repaired_count`, entre outras.
- **Pós-reparo (resgatado)**: aproveita as predições resgatadas pelo reparo;
  permanece como erro apenas o irrecuperável (`predicted_label = -1`). Chaves
  com prefixo `rescued_` (`rescued_error_rate`, `rescued_accuracy`, ...).

A interpretação é a seguinte: houve taxa de erro X na saída bruta; ao reparar
esses erros com heurísticas externas, os resultados passam a ser Y. A classe
positiva é PHISHING (1) e o relatório é ordenado por `f1_score`.

## Pré-requisitos

- **Python 3.10+**.
- **Chave da API da Anthropic** para executar o Claude.
- **Docker Model Runner** ativo em `http://localhost:12434` para executar os
  modelos locais. O DMR mantém um modelo por vez em memória e realiza o swap
  ao receber requisição para outro modelo; por isso o benchmark executa os
  modelos sequencialmente e chama `warm_up()` para pré-carregar cada um. Sem o
  pré-carregamento, o tempo de carga do modelo seria contabilizado como
  latência do primeiro e-mail.

## Instalação

```bash
pip install -r requirements.txt
```

Copie o arquivo de exemplo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Em seguida, abra o `.env` e substitua o valor pela sua chave da API:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Por fim, coloque o dataset em `dataset/` e configure o caminho correspondente
em [config.py](config.py).

## Configuração

Toda a configuração fica em [config.py](config.py). Principais pontos:

| Opção | Descrição |
| --- | --- |
| `CLAUDE_MODEL` | Modelo da Anthropic a ser testado. |
| `RUN_CLAUDE` | Liga/desliga o teste do Claude. |
| `LOCAL_MODELS` | Mapeia *nome amigável: tag do modelo no DMR*. |
| `DMR_BASE_URL` | Endpoint do Docker Model Runner. |
| `DATASET_PATH` | Caminho do CSV de e-mails. **Deve apontar para o dataset real.** |
| `PHISHING_LIKELIHOOD_THRESHOLDS` | Ponto de corte por modelo, aplicado pelo scorer sobre `phishing_likelihood` (fallback na chave `"Default"`). |
| `ENABLE_LLM_LOGGING` | Se `True`, grava um CSV por modelo em `results/llm_logs/`. |
| `SKIP_COMPLETED_MODELS` | Se `True`, pula modelos que já têm `metrics_<modelo>.json` (retomada por modelo). |
| `RESUME_PARTIAL_MODEL` | Se `True`, retoma por e-mail um modelo interrompido no meio: reaproveita os e-mails já gravados no log parcial (exceto o último, que é re-testado) e continua no mesmo CSV. Requer `ENABLE_LLM_LOGGING = True`. |
| `MAX_TOKENS` / `TEMPERATURE` | Parâmetros de geração compartilhados por todos os modelos. |
| `TRUNCATE_EMAIL_BODY` / `EMAIL_BODY_MAX_CHARS` | Trunca o corpo do e-mail. |
| `MAX_RETRIES` / `RETRY_DELAY` | Política de retentativa em caso de falha de rede. |
| `USE_CONSTRAINED_DECODING` | Se `True`, restringe a saída ao schema JSON compartilhado via decodificação estruturada. |

### Dataset

O CSV precisa ter as colunas obrigatórias:

| Coluna | Descrição |
| --- | --- |
| `id` | Identificador do e-mail. |
| `subject` | Assunto. |
| `body` | Corpo do e-mail. |
| `label` | `0` = legítimo, `1` = phishing. |

#### Montagem e reconstrução do dataset

O dataset usado no benchmark é uma amostra balanceada 50/50 (phishing/legítimo)
construída a partir de duas fontes: o **Enron Corpus** (e-mails legítimos) e
coleções de phishing (entre elas o **Nazario Phishing Corpus**). Os CSVs de
origem são grandes e não são versionados; o repositório guarda o dataset final
e a receita para reconstruí-lo de forma determinística:

| Arquivo | Versionado? | Papel |
| --- | --- | --- |
| `dataset/legitimo/emails.csv` | Não (baixar) | Corpus Enron bruto (colunas `file, message`). |
| `dataset/phishing/*.csv` | Não (baixar) | CSVs de phishing (`phishing.csv`, `phishing_2024.csv`, `Nazario_5.csv`). |
| `dataset/dataset_sample_ids.json` | Sim | Ids das amostras sorteadas, semente e metadados. |
| `dataset/dataset_provenance.csv` | Sim | Origem de cada e-mail do dataset final (gerado por `build_provenance.py`). |
| `dataset/emails_dataset.csv` | Sim | Dataset final usado no benchmark. |

Passos para reconstruir:

1. Baixe os CSVs de origem em
   [Kaggle: Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data)
   e salve-os em `dataset/legitimo/` e `dataset/phishing/`.

2. Reconstrua o dataset a partir dos ids salvos (caminho recomendado, pois
   reproduz exatamente a amostra original):

   ```bash
   python dataset/prepare_dataset.py --rebuild
   ```

   O script seleciona nas fontes exatamente os e-mails registrados em
   `dataset_sample_ids.json`, sem re-amostrar, o que garante a
   reprodutibilidade mesmo entre versões diferentes de bibliotecas.

3. Aponte `DATASET_PATH` em [config.py](config.py) para
   `dataset/emails_dataset.csv`.

Para gerar uma **nova** amostra (ex.: outro tamanho ou semente) em vez de
reconstruir a existente, o que sobrescreve os ids e metadados salvos:

```bash
python dataset/prepare_dataset.py --samples 2000 --seed 42
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

Com isso, `python main.py` já inclui o novo modelo no benchmark. Antes de um
benchmark completo, recomendam-se os passos a seguir.

### Passos opcionais (recomendados antes de um benchmark completo)

- **Smoke test**: confirma que o modelo responde e que o formato é válido.
  Abra [test_model.py](tests/test_model.py), ajuste a constante para o nome do
  seu modelo (definido no passo 2) e execute:

  ```bash
  python tests/test_model.py
  ```

- **Consistência de formato JSON**: útil para modelos pequenos, que tendem a
  quebrar o formato em prompts grandes. Executa o mesmo prompt N vezes (lê o
  e-mail de `tests/long_email.txt`):

  ```bash
  python tests/test_output.py
  ```

- **Parser / reparo de JSON**: teste offline (não usa DMR nem a API) que cobre
  o parsing e os fallbacks de reparo, incluindo aspas internas não escapadas e
  JSON truncado:

  ```bash
  python tests/test_parser.py
  ```

Para remontar o relatório comparativo juntando os `metrics_*.json` já
existentes (sem executar nenhum modelo):

```bash
python metrics.py --rebuild
```

Após alterar um ponto de corte em `config.py`, re-pontue todos os logs e
recalcule as métricas (sem re-inferir nenhum modelo):

```bash
python apply_thresholds.py            # re-pontua e recalcula
python apply_thresholds.py --dry-run  # apenas mostra o que mudaria
```

> Os scripts em `tests/` (`test_model.py`, `test_output.py`, `test_parser.py`,
> `test_scorer.py`, `test_resume.py`) são scripts executáveis standalone, não
> testes pytest; execute-os a partir da raiz do projeto. Não há framework de
> testes nem linter configurados.

## Saídas

Tudo é gravado em `results/` (gitignored):

- `results/reports/metrics_<modelo>.json`: métricas de cada modelo.
- `results/reports/comparison_report.csv`: comparativo final, ordenado por F1.
- `results/llm_logs/<modelo>_<timestamp>.csv`: registro de cada e-mail, com
  todas as informações relacionadas a cada entrada (se
  `ENABLE_LLM_LOGGING = True`). Inclui a coluna `was_repaired` (`True` quando a
  resposta só pôde ser lida após reparo heurístico; conta como erro na faixa
  bruta, mas a predição é preservada).

## Fluxo / arquitetura

`main.py` orquestra → `load_dataset()` → para cada modelo: `run_model_tests()`
→ `apply_thresholds.score_run()` → `compute_metrics()` →
`save_comparison_report()`.

| Arquivo | Responsabilidade |
| --- | --- |
| [main.py](main.py) | Orquestra o benchmark; executa os modelos sequencialmente. |
| [config.py](config.py) | Fonte única de configuração. |
| [dataset/dataset.py](dataset/dataset.py) | `load_dataset()` carrega e valida o CSV. |
| [prompt.py](prompt.py) | Prompt compartilhado por todos os modelos; `build_prompt()` trunca o corpo. |
| [output_parser.py](output_parser.py) | `parse_response()` extrai os campos estruturados do JSON (não decide o rótulo). |
| [apply_thresholds.py](apply_thresholds.py) | Scorer: aplica o corte por modelo e preenche o veredicto. |
| [models/claude_client.py](models/claude_client.py) | `ClaudeClient`, cliente da API Claude. |
| [models/dmr_client.py](models/dmr_client.py) | `DmrClient`, que usa a lib `openai` apontada para o endpoint local do DMR; tem `warm_up()`. |
| [model_runner.py](model_runner.py) | `run_model_tests()` itera o dataset e coleta as respostas. |
| [logger.py](logger.py) | `BenchmarkLogger` grava um CSV por modelo. |
| [metrics.py](metrics.py) | Calcula métricas (sklearn) e gera o relatório comparativo. |

Os dois clientes são intercambiáveis: ambos expõem
`classify(system_prompt, user_prompt) -> dict` com as chaves `raw_response`,
`elapsed_seconds`, `input_tokens`, `output_tokens` e `error`.

## Tratamento de erros

Os clientes tratam erros internamente e retornam `error` no dict, sem levantar
exceção por e-mail. Um modelo local que falha não derruba o benchmark, pois
`main.py` segue para o próximo; uma falha no Claude, por outro lado, encerra a
execução com `sys.exit(1)`.

## Retomada de benchmarks interrompidos

Como cada e-mail é gravado no CSV de log imediatamente (append a cada linha,
não ao fim da execução), um benchmark interrompido pode ser retomado sem
refazer tudo. Há dois níveis, que funcionam em conjunto:

- **Por modelo** (`SKIP_COMPLETED_MODELS = True`): pula modelos que já têm
  `metrics_<modelo>.json` (esse JSON só é escrito quando o modelo conclui
  todas as amostras).
- **Por e-mail** (`RESUME_PARTIAL_MODEL = True`): para o modelo interrompido
  no meio (sem `metrics_*.json`, mas com log parcial), reaproveita os e-mails
  já gravados em `results/llm_logs/<modelo>_<timestamp>.csv`, re-testa apenas
  o último (para garantir que a última linha não ficou gravada pela metade) e
  continua gravando no mesmo arquivo de log. As métricas finais cobrem o
  conjunto completo.

  Requer `ENABLE_LLM_LOGGING = True`. Para forçar um modelo a executar do
  zero, apague o CSV dele em `results/llm_logs/`.

Para retomar, basta reexecutar `python main.py`: o programa detecta o estado e
continua de onde parou.
