"""
Testa o scorer (apply_thresholds.py): a verificação de ACERTO, desacoplada da
inferência.

Cobre a decisão de rótulo a partir da nota numérica e do corte POR MODELO:
  - phishing_likelihood >= corte do modelo => PHISHING (1), senão LEGÍTIMO (0);
  - limiar inclusivo (nota == corte => PHISHING);
  - sem nota utilizável OU erro de chamada => predicted_label = -1 (erro);
  - is_error combina erro de chamada, reparo de JSON e ausência de nota;
  - idempotência (re-pontuar não muda nada);
  - lookup do corte por modelo, com fallback para "Default".

Este teste é AUTOCONTIDO e offline (não usa DMR nem a API do Claude).

Uso
---
    python tests/test_scorer.py
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from apply_thresholds import threshold_for, score_record, score_records
from config import PHISHING_LIKELIHOOD_THRESHOLDS, CLAUDE_MODEL

PASSED = []
FAILED = []


def check(cond: bool, msg: str):
    (PASSED if cond else FAILED).append(msg)
    print(f"  [{'OK  ' if cond else 'FALHA'}] {msg}")


def rec(likelihood="", true_label=1, repaired=False, error=""):
    """Monta um registro mínimo com os fatos que o scorer lê."""
    return {
        "phishing_likelihood": likelihood,
        "true_label":          true_label,
        "was_repaired":        repaired,
        "error_message":       error,
    }


def test_threshold_lookup():
    print("\n[1] Lookup do corte por modelo (com fallback Default)")
    check(threshold_for("gemma3_4b_qat") == 80.0, "gemma3_4b_qat => 80.0")
    check(threshold_for(CLAUDE_MODEL) == 76.5, "Claude => 76.5")
    check(threshold_for("modelo_inexistente") == PHISHING_LIKELIHOOD_THRESHOLDS["Default"],
          "modelo desconhecido => Default")


def test_above_and_below_threshold():
    print("\n[2] Acima do corte => PHISHING; abaixo => LEGÍTIMO")
    r = score_record(rec(likelihood=90, true_label=1), 80)
    check(r["predicted_label"] == 1 and r["predicted_label_text"] == "PHISHING", "90 >= 80 => PHISHING")
    check(r["is_correct"] is True, "is_correct=True (1 == true_label 1)")

    r = score_record(rec(likelihood=70, true_label=0), 80)
    check(r["predicted_label"] == 0 and r["predicted_label_text"] == "LEGÍTIMO", "70 < 80 => LEGÍTIMO")
    check(r["is_correct"] is True, "is_correct=True (0 == true_label 0)")


def test_threshold_boundary_inclusive():
    print("\n[3] Limiar é inclusivo: nota == corte => PHISHING")
    r = score_record(rec(likelihood=80, true_label=1), 80)
    check(r["predicted_label"] == 1, "80 >= 80 classifica como PHISHING")
    # corte fracionário: 76 < 76.5 (LEGÍTIMO); 77 >= 76.5 (PHISHING)
    check(score_record(rec(likelihood=76), 76.5)["predicted_label"] == 0, "76 < 76.5 => LEGÍTIMO")
    check(score_record(rec(likelihood=77), 76.5)["predicted_label"] == 1, "77 >= 76.5 => PHISHING")


def test_no_likelihood_is_error():
    print("\n[4] Sem nota numérica utilizável => erro (-1)")
    r = score_record(rec(likelihood="", true_label=1), 80)
    check(r["predicted_label"] == -1, "predicted_label=-1 sem nota")
    check(r["predicted_label_text"] == "INVÁLIDO", "texto INVÁLIDO sem nota")
    check(r["is_correct"] is False, "is_correct=False sem nota")


def test_client_error_is_error():
    print("\n[5] Erro de chamada => -1 com texto ERRO")
    r = score_record(rec(likelihood="", true_label=1, error="timeout"), 80)
    check(r["predicted_label"] == -1, "predicted_label=-1 em erro de chamada")
    check(r["predicted_label_text"] == "ERRO", "texto ERRO")


def test_scorer_preserves_is_error():
    print("\n[6] O scorer NÃO recalcula is_error (preserva correção manual)")
    r = rec(likelihood=90, true_label=1)
    r["is_error"] = "valor_intacto"   # sentinela: o scorer não deve tocar nisto
    score_record(r, 80)
    check(r["predicted_label"] == 1, "predicted_label preenchido a partir da nota")
    check(r["is_correct"] is True, "is_correct preenchido")
    check(r["is_error"] == "valor_intacto", "is_error preservado (não tocado pelo scorer)")


def test_float_string_likelihood():
    print("\n[7] Nota em formato float ('85.0', dos modelos locais) é aceita")
    r = score_record(rec(likelihood="85.0", true_label=1), 80)
    check(r["predicted_label"] == 1, "'85.0' >= 80 => PHISHING")


def test_per_model_threshold_changes_label():
    print("\n[8] O mesmo nota muda de rótulo conforme o corte do modelo")
    g = score_record(rec(likelihood=80, true_label=1), threshold_for("gemma3_4b_qat"))   # corte 80.0
    q = score_record(rec(likelihood=80, true_label=1), threshold_for("qwen3_4b"))         # corte 92.5
    check(g["predicted_label"] == 1, "nota 80 com corte 80 (gemma) => PHISHING")
    check(q["predicted_label"] == 0, "nota 80 com corte 92.5 (qwen) => LEGÍTIMO")


def test_idempotent():
    print("\n[9] Pontuar duas vezes não muda o resultado")
    r = rec(likelihood=90, true_label=1)
    score_records([r], 80)
    primeiro = dict(r)
    score_records([r], 80)
    check(r == primeiro, "registro idêntico após a 2ª pontuação")


def main():
    print(f"{'=-'*30 + '='}")
    print("Teste do scorer (verificação de acerto desacoplada)")
    print(f"{'=-'*30 + '='}")

    test_threshold_lookup()
    test_above_and_below_threshold()
    test_threshold_boundary_inclusive()
    test_no_likelihood_is_error()
    test_client_error_is_error()
    test_scorer_preserves_is_error()
    test_float_string_likelihood()
    test_per_model_threshold_changes_label()
    test_idempotent()

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
