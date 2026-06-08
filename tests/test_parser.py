"""
Testa o parsing e o reparo heurístico de JSON em output_parser.py.

Foco especial no 4º fallback: aspas duplas literais não escapadas dentro de
strings (erro comum de modelos pequenos), que devem ser RESGATADAS, porém
marcadas como reparo (repaired=True) — sinal de que a saída bruta não era um
JSON válido.

O parser NÃO decide rótulo nem acerto (isso é do scorer, ver tests/test_scorer.py):
aqui só verificamos a EXTRAÇÃO dos campos (classification_text, phishing_likelihood,
indicators) e o reparo.

Este teste é AUTOCONTIDO e offline (não usa DMR nem a API do Claude).

Uso
---
    python tests/test_parser.py
"""
import os
import sys

# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# O console do Windows (cp1252) não imprime alguns caracteres usados nos
# relatórios (ex.: '─'); força UTF-8 para o teste rodar em qualquer terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from output_parser import parse_response

PASSED = []
FAILED = []


def check(cond: bool, msg: str):
    (PASSED if cond else FAILED).append(msg)
    print(f"  [{'OK  ' if cond else 'FALHA'}] {msg}")


# Caso exato relatado: aspas duplas internas não escapadas em "indicators".
INNER_QUOTES = '''{
  "classification": "phishing",
  "phishing_likelihood": 85,
  "indicators": [
    "Suspicious link in the top video URL: http://www.cnn.com/video/partners/email/index.html?url=/video/us/2008/07/31/moos.montauk.monster.cnn",
    "Urgent, threatening language in the top video description: 'Is it a devil dog? Is it a turtle? Is it the Montauk Monster?'",
    "Request for sensitive information in the top video description: 'CNN's Jeanne Moos asks, "what is this thing?"'",
    "Suspicious sender information: The email claims to be from CNN.com, but the sender address is not explicitly provided and may be different from the expected domain."
  ]
}'''


def test_clean_json_not_repaired():
    print("\n[1] JSON limpo é extraído e NÃO marcado como reparado")
    r = parse_response('{"classification": "legitimate", '
                       '"phishing_likelihood": 5, "indicators": ["ok"]}')
    check(r["classification_text"] == "LEGÍTIMO", "classification_text=LEGÍTIMO")
    check(r["phishing_likelihood"] == 5, "phishing_likelihood=5")
    check(r["repaired"] is False, "repaired=False para JSON limpo")
    check(r["parse_note"] == "", "sem parse_note")


def test_json_in_prose_not_repaired():
    print("\n[2] JSON bem-formado cercado de texto NÃO conta como reparo")
    raw = 'Sure! Here is the result:\n{"classification": "phishing", ' \
          '"phishing_likelihood": 99, "indicators": ["x"]} Hope it helps.'
    r = parse_response(raw)
    check(r["classification_text"] == "PHISHING", "extraiu classification_text=PHISHING do meio do texto")
    check(r["phishing_likelihood"] == 99, "extraiu phishing_likelihood=99")
    check(r["repaired"] is False, "repaired=False (apenas extração, não reparo)")


def test_truncated_json_repaired():
    print("\n[3] JSON truncado é resgatado e marcado como reparo")
    raw = '{"classification": "phishing", "phishing_likelihood": 80, "indicators": ["corte no meio'
    r = parse_response(raw)
    check(r["phishing_likelihood"] == 80, "resgatou phishing_likelihood=80 de JSON truncado")
    check(r["classification_text"] == "PHISHING", "resgatou classification_text=PHISHING")
    check(r["repaired"] is True, "repaired=True para JSON truncado")


def test_inner_quotes_repaired():
    print("\n[4] Aspas internas não escapadas são resgatadas (caso relatado)")
    r = parse_response(INNER_QUOTES)
    check(r["classification_text"] == "PHISHING", "resgatou classification_text=PHISHING")
    check(r["phishing_likelihood"] == 85, "resgatou phishing_likelihood=85")
    check(len(r["indicators"]) == 4, f"resgatou os 4 indicators (tem: {len(r['indicators'])})")
    check(r["repaired"] is True, "repaired=True (saída bruta era JSON inválido)")
    contem_aspas = any('"what is this thing?"' in x for x in r["indicators"])
    check(contem_aspas, "preservou o texto com as aspas internas no indicator")


def test_irreparable_returns_none():
    print("\n[5] Lixo sem JSON retorna campos None, sem marcar reparo")
    r = parse_response("desculpe, não consigo responder a isso")
    check(r["classification_text"] is None, "classification_text=None para resposta sem JSON")
    check(r["phishing_likelihood"] is None, "phishing_likelihood=None para resposta sem JSON")
    check(r["repaired"] is False, "repaired=False quando nada foi resgatado")
    check(r["parse_note"] == "JSON inválido ou ausente", "parse_note adequada")


def test_invalid_likelihood_is_none():
    print("\n[6] phishing_likelihood não-inteiro é registrado como inválido (None)")
    r = parse_response('{"classification": "phishing", '
                       '"phishing_likelihood": "n/a", "indicators": ["x"]}')
    check(r["phishing_likelihood"] is None, "phishing_likelihood=None quando a nota é inválida")
    check(r["classification_text"] == "PHISHING", "classification_text ainda é extraída")
    check("phishing_likelihood inválido" in r["parse_note"], "registrou a nota inválida em parse_note")


def main():
    print(f"{'=-'*30 + '='}")
    print("Teste do parser / reparo heurístico de JSON")
    print(f"{'=-'*30 + '='}")

    test_clean_json_not_repaired()
    test_json_in_prose_not_repaired()
    test_truncated_json_repaired()
    test_inner_quotes_repaired()
    test_irreparable_returns_none()
    test_invalid_likelihood_is_none()

    print(f"\n{'=-'*30 + '='}")
    print(f"Resultado: {len(PASSED)} OK, {len(FAILED)} falha(s)")
    print(f"{'=-'*30 + '='}")
    if FAILED:
        for msg in FAILED:
            print(f"  FALHOU: {msg}")
        raise SystemExit(1)
    print("Todos os testes passaram.")


if __name__ == "__main__":
    main()
