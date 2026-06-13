"""
Valida a QUALIDADE e a VALIDADE de um dataset antes de rodar o benchmark.

Motivação: o pipeline trata `label=1` como "phishing" e `label=0` como
"legítimo", mas nada garante que os dados realmente representem isso. Um corpus
clássico de ham+spam (spam de farmácia/adulto + threads de listas técnicas)
passa em `load_dataset()` sem erro e produz um benchmark que MEDE OUTRA
COISA (spam x tráfego de lista), invalidando a conclusão sobre phishing.

Este script sinaliza esse tipo de problema com heurísticas transparentes:
  - distribuição/balanceamento dos rótulos;
  - % de `label=1` com aparência de phishing/golpe (isca + link/ação);
  - caracterização do que o `label=1` realmente é (farmácia, adulto, etc.);
  - % de `label=0` que parece tráfego de lista técnica (pouco representativo
    de "e-mail legítimo");
  - higiene geral: duplicatas, corpos/assuntos vazios, e-mails muito curtos;
  - amostras para inspeção visual (label=1 que NÃO parece phishing; label=0).

As heurísticas são RUIDOSAS por natureza: servem para levantar suspeita, não
para rotular. A saída é um resumo PASS / AVISO / FALHA, e o processo termina com
código != 0 se algum problema crítico for detectado (para servir de "portão"
antes de um run).

Este teste é AUTOCONTIDO e offline (não usa DMR nem a API do Claude).

Uso
---
    python tests/validate_dataset.py                 # usa config.DATASET_PATH
    python tests/validate_dataset.py outro/arq.csv   # valida outro CSV
"""
import os
import re
import sys

# adiciona a raiz do projeto ao sys.path para encontrar os módulos.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# O console do Windows (cp1252) não imprime alguns caracteres usados nos
# relatórios (ex.: '─'); força UTF-8 para o teste rodar em qualquer terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DATASET_PATH
from dataset.dataset import load_dataset

# =-=-= Limiares de alerta =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=
# Abaixo destes valores o dataset é SUSPEITO de não medir o que diz medir.
MIN_PHISHING_LIKE_RATIO = 0.50  # < isto em label=1 => AVISO (pouca cara de phishing)
FAIL_PHISHING_LIKE_RATIO = 0.20  # < isto => FALHA (label=1 não é phishing)
MAX_TECH_LIST_RATIO = 0.50  # > isto em label=0 => AVISO (legítimo = lista técnica)
MIN_MINORITY_CLASS_RATIO = 0.20  # classe minoritária abaixo disto => AVISO
MAX_DUP_RATIO = 0.02  # > 2% de duplicatas exatas => AVISO
SHORT_EMAIL_CHARS = 25  # assunto+corpo abaixo disto é "muito curto"

# =-=-= Heurísticas (regex) =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=-=-=-=
# Iscas fortes típicas de phishing/golpe (credencial, conta, cobrança, urgência).
_LURE = re.compile(
    r"\b(account|verify|confirm|suspend(ed)?|password|log\s?in|sign\s?in|"
    r"billing|invoice|payment|paypal|ebay|amazon|bank|wire transfer|"
    r"security alert|update your|your account|click the link|expire(d|s)?|"
    r"urgent|immediately|within 24|locked|unauthorized|refund|tax|irs)\b",
    re.I,
)
# Link / chamada para clicar (sozinho é fraco; combinado com ação, é sinal).
_URL = re.compile(r"https?://|www\.|\.php\b|click here", re.I)
# Verbo de ação que, junto de um link, caracteriza a isca.
_ACTION = re.compile(
    r"\b(click|verify|confirm|log\s?in|sign\s?in|download|update|pay|open|review)\b",
    re.I,
)

# Caracterização do que o label=1 REALMENTE é (categorias de spam comuns).
_CATEGORIES = {
    "farmácia/saúde": re.compile(
        r"\b(viagra|cialis|pharmac|medication|meds|pills?|erectile|"
        r"libido|weight loss|prescription)\b", re.I),
    "adulto":         re.compile(
        r"\b(xxx|porn|nude|sex|nudes|webcam|girls?)\b", re.I),
    "phishing/golpe": _LURE,
    "tem link":       _URL,
}

# Marcadores de tráfego de lista de discussão / técnico (pouco representativo
# de "e-mail legítimo" genérico: faltam transacionais, corporativos, etc.).
_TECH_LIST = re.compile(
    r"^\s*re:|wrote:|^\s*>|begin pgp|mailing list|unsubscribe|"
    r"\[[a-z0-9._-]+\]|--\s*\n|listinfo|/dev\b|patch", re.I | re.M)


def _is_phishing_like(text: str) -> bool:
    """Heurística: tem isca forte, OU um link combinado com verbo de ação."""
    tem_isca = bool(_LURE.search(text))
    tem_link_com_acao = bool(_URL.search(text)) and bool(_ACTION.search(text))
    return tem_isca or tem_link_com_acao


# =-=-= Acumuladores do resultado =-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-==-=-=
WARNINGS = []
FAILURES = []


def warn(msg: str):
    WARNINGS.append(msg)
    print(f"  [AVISO] {msg}")


def fail(msg: str):
    FAILURES.append(msg)
    print(f"  [FALHA] {msg}")


def ok(msg: str):
    print(f"  [OK]    {msg}")


def _pct(part: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{(100 * part / total):.1f}%"


def _texts(df):
    """Série 'assunto + corpo' para análise."""
    return (df["subject"] + " " + df["body"])


def check_balance(df):
    print("\n[1] Balanceamento dos rótulos")
    total = len(df)
    n1 = int((df["label"] == 1).sum())
    n0 = int((df["label"] == 0).sum())
    print(f"      total={total}  |  phishing(1)={n1} ({_pct(n1, total)})  "
          f"|  legítimo(0)={n0} ({_pct(n0, total)})")
    minority = min(n1, n0)
    if total == 0:
        fail("dataset vazio")
    elif minority / total < MIN_MINORITY_CLASS_RATIO:
        warn(f"classe minoritária com apenas {_pct(minority, total)} "
             f"(< {MIN_MINORITY_CLASS_RATIO:.0%}); métricas ficam instáveis")
    else:
        ok("classes razoavelmente balanceadas")


def check_phishing_like(df):
    print("\n[2] label=1 realmente parece phishing?")
    pos = df[df["label"] == 1]
    n = len(pos)
    if n == 0:
        warn("não há amostras com label=1")
        return
    like = _texts(pos).map(_is_phishing_like)
    n_like = int(like.sum())
    ratio = n_like / n
    print(f"      {n_like}/{n} ({_pct(n_like, n)}) têm aparência de phishing/golpe "
          f"(isca forte, ou link + ação)")
    if ratio < FAIL_PHISHING_LIKE_RATIO:
        fail(f"apenas {_pct(n_like, n)} do label=1 parece phishing; o rótulo "
             f"'phishing' provavelmente NÃO corresponde ao conteúdo")
    elif ratio < MIN_PHISHING_LIKE_RATIO:
        warn(f"apenas {_pct(n_like, n)} do label=1 parece phishing; o resto parece "
             f"spam genérico (ver categorias abaixo)")
    else:
        ok(f"{_pct(n_like, n)} do label=1 tem aparência de phishing")

    # Caracteriza o que o label=1 de fato é.
    print("      composição do label=1 (categorias podem se sobrepor):")
    texts = _texts(pos)
    for nome, rgx in _CATEGORIES.items():
        c = int(texts.map(lambda t: bool(rgx.search(t))).sum())
        print(f"        - {nome:<16} {c:>5} ({_pct(c, n)})")


def check_legit_representativeness(df):
    print("\n[3] label=0 representa 'e-mail legítimo' de verdade?")
    neg = df[df["label"] == 0]
    n = len(neg)
    if n == 0:
        warn("não há amostras com label=0")
        return
    techy = _texts(neg).map(lambda t: bool(_TECH_LIST.search(t)))
    n_techy = int(techy.sum())
    print(f"      {n_techy}/{n} ({_pct(n_techy, n)}) parecem tráfego de lista "
          f"técnica/discussão (Re:, citações '>', PGP, [lista], etc.)")
    if n_techy / n > MAX_TECH_LIST_RATIO:
        warn(f"{_pct(n_techy, n)} do label=0 é tráfego de lista técnica, "
             f"pouco representativo de e-mail legítimo genérico "
             f"(faltam transacionais, corporativos, newsletters)")
    else:
        ok("label=0 não parece dominado por tráfego de lista técnica")


def check_hygiene(df):
    print("\n[4] Higiene geral")
    total = len(df)

    empty_body = int((df["body"].str.strip() == "").sum())
    empty_subj = int((df["subject"].str.strip() == "").sum())
    if empty_body:
        warn(f"{empty_body} e-mail(s) com corpo vazio")
    if empty_subj:
        warn(f"{empty_subj} e-mail(s) com assunto vazio")
    if not (empty_body or empty_subj):
        ok("sem assuntos/corpos vazios")

    lengths = (df["subject"].str.len() + df["body"].str.len())
    short = int((lengths < SHORT_EMAIL_CHARS).sum())
    if short:
        warn(f"{short} e-mail(s) muito curto(s) (< {SHORT_EMAIL_CHARS} chars), "
             f"com pouco conteúdo para classificar")
    else:
        ok(f"nenhum e-mail abaixo de {SHORT_EMAIL_CHARS} chars")

    dup = int((df["subject"] + "\x00" + df["body"]).duplicated().sum())
    if dup and dup / total > MAX_DUP_RATIO:
        warn(f"{dup} duplicata(s) exata(s) de assunto+corpo "
             f"({_pct(dup, total)}), o que pode inflar métricas")
    elif dup:
        ok(f"{dup} duplicata(s) ({_pct(dup, total)}, dentro do tolerável)")
    else:
        ok("sem duplicatas exatas")

    dup_id = int(df["id"].duplicated().sum())
    if dup_id:
        warn(f"{dup_id} id(s) repetido(s); ids deveriam ser únicos")


def show_samples(df):
    print("\n[5] Amostras para inspeção visual")
    pos = df[df["label"] == 1]
    not_phish = pos[~_texts(pos).map(_is_phishing_like)]
    print(f"\n   label=1 que NÃO parece phishing ({len(not_phish)} no total), "
          f"suspeitos de rótulo errado:")
    for _, r in not_phish.head(5).iterrows():
        print(f"     · id {r['id']} | {r['subject'][:60]!r}")
        print(f"          {r['body'][:90]!r}")

    print("\n   label=0 (amostra do 'legítimo'):")
    for _, r in df[df["label"] == 0].head(5).iterrows():
        print(f"     · id {r['id']} | {r['subject'][:60]!r}")
        print(f"          {r['body'][:90]!r}")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = DATASET_PATH
    print(f"{'=-'*30 + '='}")
    print(f"Validação de dataset: {path}")
    print(f"{'=-'*30 + '='}")

    try:
        df = load_dataset(path)
    except FileNotFoundError:
        print(f"\n[FALHA] arquivo não encontrado: {path}")
        raise SystemExit(2)
    except ValueError as e:
        print(f"\n[FALHA] dataset inválido: {e}")
        raise SystemExit(2)

    check_balance(df)
    check_phishing_like(df)
    check_legit_representativeness(df)
    check_hygiene(df)
    show_samples(df)

    print(f"\n{'=-'*30 + '='}")
    print(f"Resumo: {len(FAILURES)} falha(s), {len(WARNINGS)} aviso(s)")
    print(f"{'=-'*30 + '='}")
    if FAILURES:
        print("Problemas CRÍTICOS (o benchmark provavelmente mediria outra coisa):")
        for m in FAILURES:
            print(f"  - {m}")
        raise SystemExit(1)
    if WARNINGS:
        print("Avisos (revise antes de confiar nos resultados):")
        for m in WARNINGS:
            print(f"  - {m}")
        # Avisos não falham o portão: são para julgamento humano.
    else:
        print("Nenhum problema detectado pelas heurísticas.")


if __name__ == "__main__":
    main()
