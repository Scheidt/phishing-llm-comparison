const TEXT = "Your input text goes here.";
const THRESHOLD = 0.7; // Exemplo: 70% de palavras em inglês

const ENGLISH_WORDS_REGEX = /\b(the|be|to|of|and|a|in|that|have|it|for|not|on|with|he|as|you|do|at|this|but|his|by|from|they|we|say|her|she|or|an|will|my|one|all|would|there|their|what|so|up|out|if|about|who|get|which|go|me|when|make|can|like|time|no|just|him|know|take|people|into|year|your|good|some|could|them|see|other|than|then|now|look|only|come|its|over|think|also|back|after|use|two|how|our|work|first|well|way|even|new|want|because|any|these|give|day|most|us|is|was|are|were|been|has|had|did|does|said|may|might|shall|should|must|need|used|made|came|went|got|put|set|run|let|ask|seem|feel|try|leave|call|keep|hold|bring|begin|show|hear|play|turn|move|live|believe|hold|allow|lead|place|stand|change|add|end|open|offer|appear|buy|wait|serve|die|send|expect|build|stay|fall|cut|reach|kill|remain|suggest|raise|pass|sell|require|report|decide|pull)\b/gi;

if (!text || typeof text !== "string") return false;

// Total de palavras no texto (tokens)
const words = text.match(/\b\w+\b/g);
if (!words || words.length === 0) return false;

const totalWords = words.length;

// Reseta o lastIndex antes de usar a regex global
ENGLISH_WORDS_REGEX.lastIndex = 0;
const matchedWords = text.match(ENGLISH_WORDS_REGEX) ?? [];

const ratio = matchedWords.length / totalWords;

console.log(`Total de palavras   : ${totalWords}`);
console.log(`Palavras em inglês  : ${matchedWords.length}`);
console.log(`Proporção           : ${(ratio * 100).toFixed(1)}%`);
console.log(`Limiar mínimo       : ${(threshold * 100).toFixed(1)}%`);

return ratio >= threshold;