/**
 * analisarVirusTotal.js
 * Analisa os resultados do VirusTotal retornados por um workflow N8N
 * e produz um relatório consolidado de segurança.
 *
 * @param {Array} vtData - Array raiz retornado pelo nó do VirusTotal no N8N
 * @returns {Object} Relatório de segurança consolidado
 */
function analisarVirusTotal(vtData) {
  // ── Configurações de limiar ──────────────────────────────────────────────
  const LIMIAR_SCORE_PERIGO = 20;   // score ≥ este valor → linkMalicioso = true
  const PESO_MALICIOSO      = 1.0;  // peso de cada detecção maliciosa
  const PESO_SUSPEITO       = 0.5;  // peso de cada detecção suspeita

  // ── Estruturas de resultado por URL ─────────────────────────────────────
  const resultadosPorURL = [];

  // A raiz do JSON N8N é um array de wrappers; a lista real está em [0].data
  const analises = vtData?.[0]?.data ?? [];

  for (const item of analises) {
    const attrs = item?.data?.attributes ?? {};
    const url   = attrs.url ?? "(URL desconhecida)";

    // URLs ainda em fila não possuem resultados por motor
    if (attrs.status === "queued") {
      resultadosPorURL.push({
        url,
        status           : "queued",
        scoreParcial     : null,
        malicioso        : false,
        suspeito         : false,
        sitesComAviso    : [],   // detecções suspeitas
        sitesQueConfirmaram: [], // detecções maliciosas
        stats            : attrs.stats ?? {},
      });
      continue;
    }

    // ── Resultados por motor para URLs já analisadas ─────────────────────
    const motores = Object.values(attrs.results ?? {});

    const sitesQueConfirmaram = motores
      .filter(m => m.category === "malicious")
      .map(m => m.engine_name);

    const sitesComAviso = motores
      .filter(m => m.category === "suspicious")
      .map(m => m.engine_name);

    const stats = attrs.stats ?? {};

    // Total de motores que emitiram algum veredicto (exclui timeout/failure)
    const totalVeredictos =
      (stats.malicious          ?? 0) +
      (stats.suspicious         ?? 0) +
      (stats.harmless           ?? 0) +
      (stats.undetected         ?? 0);

    // Score ponderado [0–100]
    const scoreParcial = totalVeredictos > 0
      ? (
          (stats.malicious ?? 0) * PESO_MALICIOSO +
          (stats.suspicious ?? 0) * PESO_SUSPEITO
        ) / totalVeredictos * 100
      : 0;

    resultadosPorURL.push({
      url,
      status              : attrs.status,
      scoreParcial        : parseFloat(scoreParcial.toFixed(2)),
      malicioso           : (stats.malicious ?? 0) > 0,
      suspeito            : (stats.suspicious ?? 0) > 0,
      sitesComAviso,
      sitesQueConfirmaram,
      stats,
    });
  }

  // ── Consolidação global ──────────────────────────────────────────────────
  const analisadas = resultadosPorURL.filter(r => r.status !== "queued");
  const emFila     = resultadosPorURL.filter(r => r.status === "queued");

  // Score global = maior score individual (pior caso)
  const scorePerigo = analisadas.length > 0
    ? Math.max(...analisadas.map(r => r.scoreParcial))
    : 0;

  const linkMalicioso =
    scorePerigo >= LIMIAR_SCORE_PERIGO ||
    analisadas.some(r => r.malicioso);

  // Somente as URLs que individualmente justificam linkMalicioso = true
  const linksMaliciosos = analisadas
    .filter(r => r.malicioso || r.scoreParcial >= LIMIAR_SCORE_PERIGO)
    .map(r => r.url);

  // Listas globais deduplicadas
  const sitesComAviso = [
    ...new Set(analisadas.flatMap(r => r.sitesComAviso)),
  ];

  const sitesQueConfirmaram = [
    ...new Set(analisadas.flatMap(r => r.sitesQueConfirmaram)),
  ];

  // URLs suspeitas: score > 0 mas abaixo do limiar, sem confirmação maliciosa
  const linksSuspeitos = analisadas
    .filter(r => r.suspeito && !r.malicioso)
    .map(r => r.url);

  // ── Objeto de retorno ────────────────────────────────────────────────────
  return {
    scorePerigo          : parseFloat(scorePerigo.toFixed(2)),
    linkMalicioso,
    linksMaliciosos,
    linksSuspeitos,
    sitesComAviso,
    sitesQueConfirmaram,
    urlsEmFila           : emFila.map(r => r.url),
    detalhesPorURL       : resultadosPorURL,
  };
}

// ── Exportação (Node / N8N Code Node) ───────────────────────────────────────
// Em um nó "Code" do N8N, substitua a linha abaixo por:
//   return analisarVirusTotal($input.all());
if (typeof module !== "undefined") {
  module.exports = { analisarVirusTotal };
}


// ── Exemplo de uso ───────────────────────────────────────────────────────────
/*
const resultado = analisarVirusTotal(dadosDoVirusTotal);
console.log(resultado);

// Saída esperada para o JSON de exemplo:
// {
//   scorePerigo: 0,
//   linkMalicioso: false,
//   linksMaliciosos: [],         // URLs que individualmente causam linkMalicioso = true
//   linksSuspeitos: [],          // URLs com ao menos um motor "suspicious" mas nenhum "malicious"
//   sitesComAviso: [],
//   sitesQueConfirmaram: [],
//   urlsEmFila: [
//     "http://site-phishing-falso.com/login?token=abc123",
//     "http://www.eicar.org/download/eicar.com"
//   ],
//   detalhesPorURL: [ ... ]
// }
*/