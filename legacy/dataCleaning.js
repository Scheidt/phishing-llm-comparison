// Padrões de Injeção de Prompt. Bom adicionar mais padrões constantemente

const injectionPatterns = [
  // Instrução direta ao modelo
  /ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?|context)/i,
  /esqueça\s+(todas?\s+)?(as\s+)?(instruções?|regras?|contexto)\s+anteriores?/i,
  /ignore\s+(todas?\s+)?(as\s+)?(instruções?|regras?|ordens?)\s+anteriores?/i,

  // Redefinição de persona
  /you\s+are\s+now\s+(a\s+)?(new\s+)?(ai|assistant|model|bot|gpt|llm)/i,
  /act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(different|new|another|unrestricted)/i,
  /você\s+é\s+agora\s+(um\s+)?(novo\s+)?(assistente|modelo|bot|ia)/i,
  /finja\s+(que\s+)?(você\s+é|ser)\s+/i,
  /pretend\s+(you\s+are|to\s+be)\s+/i,

  // Escape de contexto
  /\[system\]/i,
  /<\|im_start\|>/i,
  /<\|im_end\|>/i,
  /###\s*instruction/i,
  /\[INST\]/i,
  /<<SYS>>/i,

  // Exfiltração / revelação
  /repeat\s+(the\s+)?(above|previous|system|your)\s+(prompt|instructions?|context)/i,
  /reveal\s+(your\s+)?(system\s+)?(prompt|instructions?|context)/i,
  /what\s+(is|are)\s+your\s+(original\s+)?(instructions?|rules?|system\s+prompt)/i,
  /mostre?\s+(as?\s+)?(suas?\s+)?(instruções?|prompt\s+de\s+sistema)/i,

  // Jailbreak / desbloqueio
  /jailbreak/i,
  /dan\s+mode/i,
  /developer\s+mode/i,
  /without\s+(any\s+)?restrictions?/i,
  /sem\s+(nenhuma\s+)?restrição/i,
  /bypass\s+(your\s+)?(safety|content|filter)/i,

  // Injeção de novo contexto
  /new\s+context\s*:/i,
  /novo\s+contexto\s*:/i,
  /---\s*new\s+task/i,
  /---\s*nova\s+tarefa/i,
];

// ── 1. CAMPOS BRUTOS ──────────────────────────
const rawHtml   = $input.first().json.textHtml;
const rawText   = $input.first().json.textPlain;
const fromAddress      = $input.first().json.from;
const to        = $input.first().json.to;
const subject   = $input.first().json.subject;
const uid       = $input.first().json.attributes.uid;
const returnPath = $input.first().json.metadata['return-path'];
const received   = $input.first().json.metadata.received;
const contentType = $input.first().json.metadata['content-type'];

// ── 2. DOMÍNIOS ───────────────────────────────
const senderDomain = fromAddress.includes("@")
  ? fromAddress.split("@")[1].replace(/[>\s]/g, "").toLowerCase()
  : "";

const returnDomain = returnPath.includes("@")
  ? returnPath.split("@")[1].replace(/[>\s]/g, "").toLowerCase()
  : "";

// Domínio do IP no received (ex: "172.19.0.1")
const receivedIp = (received.match(/from\s+([\d.]+)/) || [])[1] || null;

// ── 3. EXTRAÇÃO DE LINKS ──────────────────────
//  Extrai de: href="...", src="...", url(...), texto puro
const urlRegex = /https?:\/\/[^\s"'<>)\]]+/gi;

const linksFromHtml = [...rawHtml.matchAll(urlRegex)].map(m => m[0]);
const linksFromText = [...rawText.matchAll(urlRegex)].map(m => m[0]);

// Deduplicar
const allLinks = [...new Set([...linksFromHtml, ...linksFromText])]
  .map(u => u.replace(/[.,;!?]$/, "")) // remove pontuação final solta
  .filter(u => u.length > 0);

// Domínios únicos dos links
const linkDomains = [...new Set(
  allLinks.map(u => {
    try { return new URL(u).hostname.toLowerCase(); }
    catch { return null; }
  }).filter(Boolean)
)];

// ── 4. PROMPT INJECTION DETECTION ────────────
//
//  Verifica TEXTO PURO + HTML visível + conteúdo oculto em HTML
//  (comentários, atributos alt/title/aria, style display:none, etc.)

// 4a. Strip HTML → texto visível limpo
function stripHtml(html) {
  return html
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&lt;/gi, "<").replace(/&gt;/gi, ">")
    .replace(/\s+/g, " ")
    .trim();
}

// 4b. Extrai conteúdo de atributos HTML (alt, title, aria-label, placeholder, value)
function extractHtmlAttributes(html) {
  const attrRegex = /(?:alt|title|aria-label|aria-description|placeholder|value)\s*=\s*["']([^"']{10,})["']/gi;
  return [...html.matchAll(attrRegex)].map(m => m[1]).join(" ");
}

// 4c. Extrai comentários HTML (<!-- ... -->)
function extractHtmlComments(html) {
  return [...html.matchAll(/<!--([\s\S]*?)-->/g)].map(m => m[1]).join(" ");
}

// 4d. Extrai texto em elementos ocultos (display:none, visibility:hidden, opacity:0, font-size:0)
function extractHiddenText(html) {
  const hiddenBlockRegex = /style\s*=\s*["'][^"']*(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0|font-size\s*:\s*0)[^"']*["'][^>]*>([\s\S]*?)<\/[a-z]+>/gi;
  return [...html.matchAll(hiddenBlockRegex)].map(m => stripHtml(m[1])).join(" ");
}

// 4e. Constrói corpus completo para análise
const visibleText   = stripHtml(rawHtml);
const attrText      = extractHtmlAttributes(rawHtml);
const commentText   = extractHtmlComments(rawHtml);
const hiddenText    = extractHiddenText(rawHtml);

const fullCorpus = [
  subject,
  rawText,
  visibleText,
  attrText,
  commentText,
  hiddenText
].join(" ").toLowerCase();

const injectionMatches = injectionPatterns
  .map((pattern, i) => {
    const match = fullCorpus.match(pattern);
    return match ? { patternIndex: i, sample: match[0].substring(0, 80) } : null;
  })
  .filter(Boolean);

const injectionDetected = injectionMatches.length > 0;

// Onde foi detectado (para log)
const injectionSources = [];
if (injectionDetected) {
  const checks = [
    { label: "subject",       text: subject.toLowerCase() },
    { label: "textPlain",     text: rawText.toLowerCase() },
    { label: "visibleHtml",   text: visibleText.toLowerCase() },
    { label: "htmlAttribute", text: attrText.toLowerCase() },
    { label: "htmlComment",   text: commentText.toLowerCase() },
    { label: "hiddenElement", text: hiddenText.toLowerCase() },
  ];
  for (const p of injectionPatterns) {
    for (const c of checks) {
      if (p.test(c.text) && !injectionSources.includes(c.label)) {
        injectionSources.push(c.label);
      }
    }
  }
}

// ── 5. SCORE INICIAL ──────────────────────────
//  Prompt injection é malicioso → risco ALTO imediato
let riskScore = 0;
if (injectionDetected) riskScore += 100;

// ── 6. OUTPUT ─────────────────────────────────
return {
  // Identificação
  uid,
  fromAddress,
  to,
  subject,
  senderDomain,
  returnDomain,
  receivedIp,
  contentType,

  // Conteúdo limpo
  textPlain: rawText,
  textHtml: rawHtml,
  visibleText,          // HTML sem tags, para o prompt da IA

  // Links
  links: allLinks,        // array com URLs completas
  linkDomains,            // array com domínios únicos

  // Segurança — Prompt Injection
  injectionDetected,
  injectionMatches,       // quais padrões disparam
  injectionSources,       // onde foi detectado (htmlComment, hiddenElement, etc.)

  // Score acumulado (outros nós somam aqui)
  riskScore,

  // Se injection → força risco ALTO sem passar pela IA
  forceHighRisk: injectionDetected,
};