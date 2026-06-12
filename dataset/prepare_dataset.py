"""
Montagem reprodutível do dataset do benchmark a partir das pastas de origem.

Junta duas fontes em um único dataset balanceado no formato esperado pelo
pipeline (id, from, subject, body, label):

  - LEGÍTIMOS (label=0): dataset/legitimo/emails.csv (corpus Enron), cujo CSV
    tem as colunas `file, message`, onde `message` é o e-mail RFC822 bruto
    (cabeçalhos + corpo). São extraídos `From`, `Subject` e o corpo
    (preferindo text/plain, com fallback para text/html sem tags). Como o
    arquivo é grande (~1.4 GB), a amostragem é feita por *reservoir sampling*
    em uma única passada, sem carregar tudo na memória.
  - PHISHING (label=1): montado por *preenchimento prioritário*. Primeiro
    entra a BASE (phishing.csv + phishing_2024.csv); se a cota por classe não
    for alcançada, o restante é completado com o FILL (Nazario_5.csv). Todas
    essas fontes têm `sender, subject, body` (`sender` é usado como `from`).
    Só entram linhas com label=1, e qualquer corpo já presente na base é
    descartado. (email_reserva.csv NÃO é usado: só tem `label, text`, sem
    remetente, o que deixaria `from` vazio apenas em phishing, um vazamento
    de formato.)

Em ambas as classes, todo e-mail passa por um filtro de qualidade semântica
(ver body_is_semantic / MIN_BODY_*): precisa ter texto legível suficiente para
análise por um modelo de linguagem, descartando-se HTML vazio, corpos
compostos apenas por URL, tokens codificados e fragmentos minúsculos. Tudo é
também deduplicado por corpo (sem repetidos, inclusive entre as classes).

Amostragem balanceada 50/50: `--samples` define o total (dividido por 2). Se
uma das classes não tiver amostras suficientes, o script reduz o `per_class`
ao mínimo disponível e avisa, mantendo o dataset balanceado.

Reprodutibilidade
-----------------
Os identificadores das amostras sorteadas são salvos em
dataset/dataset_sample_ids.json (os `file` do Enron para os legítimos e o hash
de conteúdo para os phishing), junto com a semente e metadados. O modo
`--rebuild` reconstrói o mesmo dataset a partir desses ids, sem re-amostrar.
Esse caminho é mais robusto que confiar apenas no --seed, pois o resultado do
sorteio pode variar entre versões de bibliotecas, enquanto a lista explícita
de ids não varia.

Uso:
    python dataset/prepare_dataset.py                  # sorteia e salva ids + metadados
    python dataset/prepare_dataset.py --rebuild        # reconstrói a partir dos ids salvos
    python dataset/prepare_dataset.py --samples 2000 --seed 7
"""
import argparse
import csv
import email
import hashlib
import json
import os
import random
import re
import sys
from email import policy

import pandas as pd

# Mensagens do Enron podem ser grandes; sobe o limite de campo do módulo csv.
csv.field_size_limit(2**31 - 1)

# Fontes de phishing, em ordem de prioridade. Todas têm `sender, subject, body`
# (remetente real, importante para o campo `from`). A BASE entra primeiro; o
# FILL completa a cota quando a base não basta. email_reserva.csv NÃO é usado:
# só tem `label, text` (sem remetente), o que deixaria `from` vazio apenas em
# phishing, um vazamento de formato.
BASE_PHISHING_FILES = ["phishing.csv", "phishing_2024.csv"]
FILL_PHISHING_FILES = ["Nazario_5.csv"]

# Filtros de qualidade semântica: o e-mail precisa ter texto legível suficiente
# para análise. Medidos sobre o TEXTO (HTML removido, espaços colapsados):
#   - MIN_BODY_CHARS:  comprimento mínimo do texto;
#   - MIN_BODY_WORDS:  nº mínimo de palavras com letras, após remover URLs e
#                      e-mails (descarta corpos compostos apenas por URL e
#                      tokens codificados tipo base64);
#   - MIN_ALPHA_RATIO: piso de proporção alfabética (descarta blobs
#                      codificados); mantido propositalmente baixo, pois
#                      e-mails legítimos com tabelas, linhas "Forwarded by" e
#                      telefones têm proporção naturalmente baixa.
MIN_BODY_CHARS = 20
MIN_BODY_WORDS = 5
MIN_ALPHA_RATIO = 0.35


def sha256_file(path: str) -> str:
    """Calcula o SHA-256 de um arquivo em blocos (sem carregá-lo inteiro na memória)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_key(subject: str, body: str) -> str:
    """Chave de conteúdo estável para deduplicar/rastrear um e-mail (hash de subject+body)."""
    h = hashlib.sha256()
    h.update(subject.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(body.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")


def readable_text(body: str) -> str:
    """Texto legível do corpo: remove tags HTML e colapsa espaços."""
    return " ".join(_TAG_RE.sub(" ", body).split())


def body_is_semantic(body: str) -> bool:
    """True se o corpo tem conteúdo textual com significado suficiente para análise.

    Aplica os três filtros (MIN_BODY_CHARS / MIN_ALPHA_RATIO / MIN_BODY_WORDS)
    sobre o texto legível, descartando e-mails sem conteúdo analisável (HTML
    vazio, corpos compostos apenas por URL, tokens codificados, fragmentos
    minúsculos).
    """
    text = readable_text(body)
    if len(text) < MIN_BODY_CHARS:
        return False
    compact = re.sub(r"\s", "", text)
    if compact and sum(c.isalpha() for c in compact) / len(compact) < MIN_ALPHA_RATIO:
        return False
    core = _URL_RE.sub(" ", text)
    words = [w for w in core.split() if any(c.isalpha() for c in w)]
    return len(words) >= MIN_BODY_WORDS


# --------------------------------------------------------------------------- #
# Legítimos (Enron)
# --------------------------------------------------------------------------- #
def _parse_enron_message(raw: str) -> tuple[str, str, str]:
    """Extrai (from, subject, body) de uma mensagem RFC822 bruta do corpus Enron."""
    msg = email.message_from_string(raw, policy=policy.default)
    sender = (msg.get("From") or "").strip()
    subject = (msg.get("Subject") or "").strip()
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        text = part.get_content() if part is not None else ""
    except Exception:
        text = ""
    body = " ".join(text.split())
    return sender, subject, body


def iter_enron_rows(path: str):
    """Itera (file, message) do CSV do Enron, uma linha por vez."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # cabeçalho: file, message
        for row in reader:
            if len(row) >= 2:
                yield row[0], row[1]


def sample_legit(path: str, per_class: int, seed: int) -> pd.DataFrame:
    """
    Sorteia `per_class` e-mails legítimos do Enron via reservoir sampling.

    Sorteia-se um reservatório maior (com folga) sobre as linhas brutas e só
    então as mensagens são parseadas e filtradas por corpo não-vazio. Dessa
    forma, evita-se parsear o corpus inteiro e ainda se garantem amostras
    úteis suficientes.
    """
    rng = random.Random(seed)
    pool_size = per_class * 3 + 500  # folga para descartar não-semânticos e duplicados

    reservoir: list[tuple[str, str]] = []
    for i, (file, raw) in enumerate(iter_enron_rows(path)):
        if i < pool_size:
            reservoir.append((file, raw))
        else:
            j = rng.randint(0, i)
            if j < pool_size:
                reservoir[j] = (file, raw)

    records = []
    seen_bodies: set[str] = set()
    for file, raw in reservoir:
        sender, subject, body = _parse_enron_message(raw)
        if not body_is_semantic(body):
            continue  # sem texto com significado, não há o que analisar
        if body in seen_bodies:
            continue  # dedup por corpo: e-mails idênticos não se repetem
        seen_bodies.add(body)
        records.append({"src_id": file, "from": sender, "subject": subject, "body": body, "label": 0})

    if len(records) < per_class:
        raise ValueError(
            f"Legítimos úteis insuficientes: {len(records)} encontrados, {per_class} pedidos. "
            f"Aumente a folga do reservatório."
        )

    rng.shuffle(records)
    return pd.DataFrame(records[:per_class])


def rebuild_legit(path: str, src_ids: list[str]) -> pd.DataFrame:
    """Reconstrói os legítimos selecionando, no Enron, as linhas cujo `file` está em src_ids."""
    wanted = set(src_ids)
    found = {}
    for file, raw in iter_enron_rows(path):
        if file in wanted:
            sender, subject, body = _parse_enron_message(raw)
            found[file] = {"src_id": file, "from": sender, "subject": subject, "body": body, "label": 0}
            if len(found) == len(wanted):
                break
    missing = wanted - set(found)
    if missing:
        sys.exit(f"ERRO: {len(missing)} ids de legítimos não encontrados no Enron (ex: {list(missing)[:3]}).")
    # Preserva a ordem salva.
    return pd.DataFrame([found[i] for i in src_ids])


# --------------------------------------------------------------------------- #
# Phishing
# --------------------------------------------------------------------------- #
def _finalize_phishing(df: pd.DataFrame) -> pd.DataFrame:
    """Anexa src_id (hash de conteúdo) e label=1 a um frame com colunas from/subject/body."""
    out = df.copy()
    out["src_id"] = [content_key(s, b) for s, b in zip(out["subject"], out["body"])]
    out["label"] = 1
    return out[["src_id", "from", "subject", "body", "label"]]


def _load_phishing_files(folder: str, names: list[str]) -> pd.DataFrame:
    """Carrega CSVs de phishing (from/subject/body): label=1, semântico, dedup por corpo.

    Deduplica por `body` (não por subject+body): templates de phishing com o
    mesmo corpo e assuntos diferentes são o *mesmo* e-mail repetido.
    """
    frames = []
    for name in names:
        df = pd.read_csv(os.path.join(folder, name))
        df = df[df["label"] == 1]
        frames.append(pd.DataFrame({
            "from": df["sender"].fillna("").astype(str),
            "subject": df["subject"].fillna("").astype(str),
            "body": df["body"].fillna("").astype(str),
        }))
    out = pd.concat(frames, ignore_index=True)
    out = out[out["body"].map(body_is_semantic)]
    out = out.drop_duplicates(subset=["body"]).reset_index(drop=True)
    return out


def sample_phishing(folder: str, per_class: int, seed: int, exclude_bodies: set[str]) -> pd.DataFrame:
    """Monta `per_class` phishing por preenchimento prioritário: base, depois fill (Nazario_5).

    `exclude_bodies` são corpos a evitar (ex.: os dos legítimos já escolhidos),
    garantindo que nenhum e-mail se repita entre as classes.
    """
    base = _load_phishing_files(folder, BASE_PHISHING_FILES)
    base = base[~base["body"].isin(exclude_bodies)].reset_index(drop=True)

    if len(base) >= per_class:
        sample = base.sample(n=per_class, random_state=seed)
        return _finalize_phishing(sample)

    need = per_class - len(base)
    fill = _load_phishing_files(folder, FILL_PHISHING_FILES)
    fill = fill[~fill["body"].isin(exclude_bodies | set(base["body"]))].reset_index(drop=True)
    if len(fill) < need:
        raise ValueError(
            f"Phishing insuficiente: base tem {len(base)}, fill tem {len(fill)} "
            f"novos, mas faltam {need} para chegar a {per_class}."
        )
    print(
        f"  base de phishing tem {len(base)}; completando {need} de "
        f"{', '.join(FILL_PHISHING_FILES)} para chegar a {per_class}.",
        file=sys.stderr,
    )
    fill_sample = fill.sample(n=need, random_state=seed)
    sample = pd.concat([base, fill_sample], ignore_index=True)
    return _finalize_phishing(sample)


def phishing_pool(folder: str) -> pd.DataFrame:
    """Pool completo de phishing candidato (base + fill), indexado por src_id.

    Usado no rebuild para reconstruir as amostras a partir dos src_ids salvos.
    A base tem prioridade: corpos do fill já presentes na base são descartados.
    """
    base = _load_phishing_files(folder, BASE_PHISHING_FILES)
    fill = _load_phishing_files(folder, FILL_PHISHING_FILES)
    fill = fill[~fill["body"].isin(set(base["body"]))]
    pool = _finalize_phishing(pd.concat([base, fill], ignore_index=True))
    return pool.drop_duplicates(subset=["src_id"]).set_index("src_id")


# --------------------------------------------------------------------------- #
# Montagem / escrita
# --------------------------------------------------------------------------- #
def write_dataset(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Atribui `id` posicional e escreve no formato do pipeline (id, from, subject, body, label)."""
    out = df.reset_index(drop=True).copy()
    out["id"] = out.index
    out = out[["id", "from", "subject", "body", "label"]]
    out.to_csv(output_path, index=False)
    return out


def build(args) -> None:
    """Sorteia uma amostra balanceada 50/50 e salva dataset, ids e metadados."""
    per_class = args.samples // 2

    # Legítimos primeiro; o phishing então evita qualquer corpo já usado por eles
    # (sem repetição entre classes).
    legit_sample = sample_legit(args.legit, per_class, args.seed)
    phishing_sample = sample_phishing(
        args.phishing_dir, per_class, args.seed, exclude_bodies=set(legit_sample["body"])
    )

    combined = pd.concat([phishing_sample, legit_sample], ignore_index=True)
    # Embaralha para que os phishing não fiquem agrupados antes dos legítimos.
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out = write_dataset(combined, args.output)

    meta = {
        "legit_source": args.legit,
        "phishing_dir": args.phishing_dir,
        "phishing_files_sha256": {
            name: sha256_file(os.path.join(args.phishing_dir, name))
            for name in BASE_PHISHING_FILES + FILL_PHISHING_FILES
        },
        "per_class": per_class,
        "total_samples": int(len(out)),
        "seed": args.seed,
        "pandas_version": pd.__version__,
        "legit_src_ids": legit_sample["src_id"].tolist(),
        "phishing_src_ids": phishing_sample["src_id"].tolist(),
    }
    with open(args.ids, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f"Dataset salvo em {args.output}: {len(out)} amostras "
        f"({int(out['label'].sum())} phishing, {int((out['label'] == 0).sum())} legítimos)"
    )
    print(f"Ids/metadados salvos em {args.ids}")


def rebuild(args) -> None:
    """Reconstrói o dataset a partir dos ids/metadados salvos (não re-amostra)."""
    with open(args.ids, encoding="utf-8") as f:
        meta = json.load(f)

    pool = phishing_pool(args.phishing_dir)
    wanted = meta["phishing_src_ids"]
    missing = [i for i in wanted if i not in pool.index]
    if missing:
        sys.exit(f"ERRO: {len(missing)} ids de phishing não encontrados (a origem mudou?).")
    phishing_sample = pool.loc[wanted].reset_index()

    legit_sample = rebuild_legit(args.legit, meta["legit_src_ids"])

    combined = pd.concat([phishing_sample, legit_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=meta["seed"]).reset_index(drop=True)
    out = write_dataset(combined, args.output)

    print(
        f"Dataset reconstruído em {args.output}: {len(out)} amostras "
        f"({int(out['label'].sum())} phishing, {int((out['label'] == 0).sum())} legítimos)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta o dataset do benchmark (legítimos Enron + phishing).")
    parser.add_argument("--legit", default="dataset/legitimo/emails.csv", help="CSV do Enron (file, message).")
    parser.add_argument("--phishing-dir", default="dataset/phishing", help="Pasta com os CSVs de phishing.")
    parser.add_argument("--output", default="dataset/emails_dataset.csv", help="Dataset de saída.")
    parser.add_argument("--ids", default="dataset/dataset_sample_ids.json", help="Arquivo de ids/metadados.")
    parser.add_argument("--samples", type=int, default=2000, help="Total de amostras (dividido 50/50).")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument("--rebuild", action="store_true", help="Reconstrói a partir dos ids salvos (não re-amostra).")
    args = parser.parse_args()

    if args.rebuild:
        rebuild(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
