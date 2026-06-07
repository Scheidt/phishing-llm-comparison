#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corrigir_formato.py
===================

Separa e trata os ERROS DE FORMATO dos logs de classificação de phishing.

A ideia é distinguir dois níveis de falha:
  - Erro de FORMATO  -> "o modelo não devolveu uma resposta utilizável"
                        (JSON quebrado, cortado, sem score, etc.)
  - Erro de CLASSIFICAÇÃO -> "o modelo respondeu certinho, mas errou o rótulo"
                        (tratado nas análises de métricas, NÃO aqui)

Este script age só sobre os erros de FORMATO. Para cada log:

  1. CONSERTO AUTOMÁTICO
     Quando a resposta crua ainda contém um `phishing_likelihood`
     recuperável (mesmo com o JSON malformado por caractere de controle,
     vírgula sobrando, array não fechado, etc.), o script extrai o score,
     reaplica o limiar de decisão e reescreve as colunas. Marca was_repaired=True.

  2. RESPOSTAS CORTADAS  (precisam ser reprocessadas pelo modelo)
     Quando NÃO há score recuperável e a resposta parece ter sido truncada,
     o script NÃO inventa um rótulo. Ele apenas junta os IDs e, ao final,
     imprime, por modelo:
            Nome do modelo: [IDs das respostas cortadas]
     Você reenvia esses e-mails ao modelo com um limite de tokens maior.

  3. CONSERTO MANUAL  (modo interativo, padrão)
     Para qualquer erro que não foi consertado sozinho, o script mostra a
     resposta crua e deixa você decidir: marcar como "cortada" (reprocessar),
     digitar um likelihood na mão, ou pular. Use --auto para desligar isso e
     fazer só o conserto automático + relatório.

Saída: para cada log de entrada gera "<nome>_corrigido.csv", preservando
exatamente as 18 colunas originais. A única marca de que houve conserto é a
coluna booleana `was_repaired` (não há classificação do "tipo" de conserto).
"""

import os
import re
import sys
import glob
import argparse
import pandas as pd

# No console do Windows o stdout costuma vir em cp1252/cp850 e embaralha os
# acentos do português. Força UTF-8 quando o terminal permitir.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ============================================================================
# CONSTANTES — descoberta automática dos logs em results/llm_logs/
# ============================================================================
# Raiz do projeto (este arquivo vive em <raiz>/analysis/), para que os caminhos
# funcionem independentemente do diretório de onde o script é chamado.
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_LOGS = os.path.join(RAIZ_PROJETO, "results", "llm_logs")

# Prefixo do nome do arquivo de cada modelo (os logs são
# "<prefixo>_<timestamp>.csv"; pegamos sempre o mais recente).
PREFIXOS_MODELO = {
    "qwen3":    "qwen3_4b",
}

# Sufixo do arquivo de saída (também usado para ignorar saídas anteriores na
# descoberta dos logs de entrada).
SUFIXO_SAIDA = "_corrigido"


def _log_mais_recente(prefixo: str):
    """Retorna o log mais recente de um modelo em DIR_LOGS, ou None.

    Ignora os próprios arquivos de saída ("*_corrigido.csv") para que uma
    re-execução não pegue a saída anterior como entrada.
    """
    padrao = os.path.join(DIR_LOGS, f"{prefixo}_*.csv")
    candidatos = [c for c in glob.glob(padrao)
                  if not c.endswith(SUFIXO_SAIDA + ".csv")]
    return max(candidatos, key=os.path.getmtime) if candidatos else None


def descobrir_logs() -> dict:
    """Mapeia nome do modelo -> caminho do log mais recente encontrado."""
    return {nome: _log_mais_recente(prefixo)
            for nome, prefixo in PREFIXOS_MODELO.items()}


# Resolvido em tempo de import; um valor None vira "log não encontrado" adiante.
LOGS = descobrir_logs()

# Limiar de decisão: phishing_likelihood >= LIMIAR  ->  phishing (1)
LIMIAR = 50

# Classe positiva e rótulos textuais usados nos logs
LABEL_PHISHING = 1
LABEL_LEGITIMO = 0
LABEL_INVALIDO = -1
TEXTO = {LABEL_PHISHING: "PHISHING", LABEL_LEGITIMO: "LEGÍTIMO", LABEL_INVALIDO: "INVÁLIDO"}

# ============================================================================
# EXTRAÇÃO / HEURÍSTICAS
# ============================================================================

def extrair_likelihood(raw: str):
    """Tenta achar um phishing_likelihood (0-100) na resposta crua."""
    if not isinstance(raw, str):
        return None
    m = re.search(r'phishing_likelihood"?\s*:\s*"?(\d{1,3})', raw)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    return None


def extrair_classification(raw: str):
    """Tenta achar a classificação textual (phishing / legitimate)."""
    if not isinstance(raw, str):
        return None
    m = re.search(r'classification"?\s*:\s*"?\s*(phishing|legitimate|leg[ií]timo)',
                  raw, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).lower()
    return LABEL_PHISHING if val.startswith("phish") else LABEL_LEGITIMO


def extrair_indicators(raw: str):
    """Tenta reextrair a lista de indicadores do bloco "indicators": [...]."""
    if not isinstance(raw, str):
        return None
    m = re.search(r'"indicators"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if not m:
        return None
    itens = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    itens = [i.strip() for i in itens if i.strip()]
    return " | ".join(itens) if itens else None


def label_por_likelihood(like: int) -> int:
    return LABEL_PHISHING if like >= LIMIAR else LABEL_LEGITIMO


def parece_cortado(raw: str) -> bool:
    """Heurística: sem score recuperável e sem fechamento -> resposta truncada."""
    if not isinstance(raw, str):
        return True
    s = raw.rstrip()
    fechou = s.endswith("}") or s.endswith("```")
    return extrair_likelihood(raw) is None and not fechou


def eh_erro_formato(linha) -> bool:
    """Um erro de formato é toda linha sinalizada como erro ou sem predição válida."""
    return bool(linha.get("is_error", False)) or int(linha.get("predicted_label", 0)) == LABEL_INVALIDO


# ============================================================================
# APLICAR CONSERTO A UMA LINHA
# ============================================================================

def aplicar_conserto(df, idx, label):
    """Reescreve as colunas de uma linha com o rótulo recuperado e marca was_repaired."""
    like = extrair_likelihood(df.at[idx, "raw_response"])
    df.at[idx, "predicted_label"] = label
    df.at[idx, "predicted_label_text"] = TEXTO[label]
    if like is not None:
        df.at[idx, "phishing_likelihood"] = float(like)
    inds = extrair_indicators(df.at[idx, "raw_response"])
    if inds:
        df.at[idx, "indicators"] = inds
    df.at[idx, "is_correct"] = (label == int(df.at[idx, "true_label"]))
    df.at[idx, "is_error"] = False
    df.at[idx, "was_repaired"] = True


# ============================================================================
# PROCESSAMENTO DE UM LOG
# ============================================================================

def processar_log(nome_modelo, caminho, interativo):
    if not caminho or not os.path.exists(caminho):
        print(f"  [aviso] log de {nome_modelo} não encontrado em {DIR_LOGS} (pulado)")
        return None

    df = pd.read_csv(caminho)

    erros_idx = [i for i in df.index if eh_erro_formato(df.loc[i])]
    total_erros = len(erros_idx)

    auto_ok, cortados, manuais = [], [], []

    # ---- passo 1: conserto automático ----
    for i in erros_idx:
        raw = df.at[i, "raw_response"]
        like = extrair_likelihood(raw)
        if like is not None:
            aplicar_conserto(df, i, label_por_likelihood(like))
            auto_ok.append(int(df.at[i, "email_id"]))
            continue
        # sem likelihood -> tenta classification textual
        cls = extrair_classification(raw)
        if cls is not None:
            aplicar_conserto(df, i, cls)
            auto_ok.append(int(df.at[i, "email_id"]))
            continue
        # irrecuperável automaticamente
        (cortados if parece_cortado(raw) else manuais).append(i)

    # ---- passo 2: conserto manual (interativo) ----
    if interativo and (cortados or manuais):
        pendentes = manuais + cortados
        print(f"\n  --- conserto manual de {nome_modelo} "
              f"({len(pendentes)} pendente(s)) ---")
        ainda_cortados = []
        for i in pendentes:
            eid = int(df.at[i, "email_id"])
            raw = str(df.at[i, "raw_response"])
            print("\n  " + "-" * 60)
            print(f"  ID {eid}  (true_label = {TEXTO[int(df.at[i,'true_label'])]})")
            print(f"  raw[:300]: {raw[:300]}")
            print(f"  raw[-120:]: {raw[-120:]}")
            resp = input("  [Enter]=marcar como cortada/reprocessar | "
                         "0-100=likelihood | l=legítimo | p=phishing | s=pular: ").strip().lower()
            if resp == "":
                ainda_cortados.append(eid)
            elif resp == "s":
                continue
            elif resp == "l":
                aplicar_conserto(df, i, LABEL_LEGITIMO)
            elif resp == "p":
                aplicar_conserto(df, i, LABEL_PHISHING)
            elif resp.isdigit() and 0 <= int(resp) <= 100:
                df.at[i, "phishing_likelihood"] = float(int(resp))
                aplicar_conserto(df, i, label_por_likelihood(int(resp)))
            else:
                print("  entrada inválida -> marcada como cortada")
                ainda_cortados.append(eid)
        cortados_ids = ainda_cortados
    else:
        cortados_ids = [int(df.at[i, "email_id"]) for i in cortados]
        manuais_ids = [int(df.at[i, "email_id"]) for i in manuais]
        cortados_ids = cortados_ids + manuais_ids  # no modo --auto, tudo que sobra vai pra lista

    # ---- salvar ----
    base, ext = os.path.splitext(caminho)
    saida = base + SUFIXO_SAIDA + ext
    df.to_csv(saida, index=False)

    return {
        "modelo": nome_modelo,
        "total_erros": total_erros,
        "auto": auto_ok,
        "cortados": cortados_ids,
        "saida": saida,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Corrige erros de formato dos logs de phishing.")
    ap.add_argument("--auto", action="store_true",
                    help="só conserto automático + relatório (sem etapa manual interativa)")
    args = ap.parse_args()
    interativo = not args.auto

    print("=" * 70)
    print("CORREÇÃO DE ERROS DE FORMATO")
    print("=" * 70)

    resultados = []
    for nome, caminho in LOGS.items():
        print(f"\n[{nome}] {caminho}")
        r = processar_log(nome, caminho, interativo)
        if r:
            print(f"  erros de formato: {r['total_erros']}  |  "
                  f"consertados automaticamente: {len(r['auto'])}  |  "
                  f"cortados (reprocessar): {len(r['cortados'])}")
            print(f"  salvo em: {r['saida']}")
            resultados.append(r)

    # -------- relatório final: respostas cortadas por modelo --------
    print("\n" + "=" * 70)
    print("RESPOSTAS CORTADAS — reenvie ao modelo com limite de tokens maior")
    print("=" * 70)
    algum = False
    for r in resultados:
        if r["cortados"]:
            algum = True
            print(f"{r['modelo']}: {sorted(r['cortados'])}")
    if not algum:
        print("(nenhuma resposta cortada pendente)")


if __name__ == "__main__":
    main()