"""
LLM-based classifier for normative acts using Azure OpenAI (gpt-5.4-mini).
Detects document type, status (VIGENTE/REVOGADO), and extracts metadata.
"""
import os
import json
import logging
import re
from typing import Dict, Any, Optional

from openai import AzureOpenAI

logger = logging.getLogger(__name__)

_AZURE_CLIENT: Optional[AzureOpenAI] = None
AZURE_LLM_MODEL = os.getenv("AZURE_LLM_MODEL", "gpt-5.4-mini")


def get_azure_client() -> Optional[AzureOpenAI]:
    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        api_key = os.getenv("AZURE_API_KEY")
        endpoint = os.getenv("AZURE_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com/")
        api_version = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
        if not api_key:
            logger.warning("AZURE_API_KEY not set, LLM classification disabled")
            return None
        _AZURE_CLIENT = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        logger.info(f"Azure classifier client initialized (model={AZURE_LLM_MODEL})")
    return _AZURE_CLIENT


def detect_strikethrough_patterns(text: str) -> bool:
    """Detect patterns that indicate revoked text."""
    revocation_patterns = [
        r'revogad[oa]',
        r'sem efeito',
        r'perde\s*efeito',
        r'perdeu\s*eficácia',
        r'revoga-se',
        r'fica\s*revogad[oa]',
        r'torna\s*sem\s*efeito',
        r'deixa\s*de\s*vigorar',
        r'ab-rogad[oa]',
        r'derrogad[oa]',
        r'\(revogad[oa]\)',
        r'REVOGAD[OA]',
    ]
    for pattern in revocation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    if '̶' in text:
        return True
    return False


SYSTEM_PROMPT = (
    "Você é um classificador especializado em atos normativos do "
    "Tribunal de Justiça de Minas Gerais (TJMG). Sua tarefa é extrair "
    "metadados estruturados de cada documento e responder APENAS em JSON válido."
)


_FILENAME_PREFIXES = {
    "re": "Resolução",
    "res": "Resolução",
    "ri": "Regimento Interno (artigo)",
    "port": "Portaria",
    "pt": "Portaria",
    "ptconj": "Portaria Conjunta",
    "port_conj": "Portaria Conjunta",
    "prov": "Provimento",
    "pr": "Provimento",
    "av": "Aviso",
    "avconj": "Aviso Conjunto",
    "av_conj": "Aviso Conjunto",
    "is": "Instrução de Serviço",
    "instr": "Instrução",
    "os": "Ordem de Serviço",
    "ord": "Ordem de Serviço",
    "er": "Emenda Regimental",
    "delib": "Deliberação",
    "enun": "Enunciado",
    "gg": "Documento administrativo (gabinete)",
}


def _filename_hint(filename: str, parent_dir: str) -> str:
    """Build a hint string describing what we can guess from filename + parent folder."""
    name = filename.lower()
    base = re.sub(r"\.[a-z]+$", "", name)
    hints: list = [f"pasta='{parent_dir}'"]
    m = re.match(r"^([a-z_]+?)(\d+)$", base)
    if m:
        prefix, digits = m.group(1), m.group(2)
        tipo_guess = _FILENAME_PREFIXES.get(prefix)
        if tipo_guess:
            hints.append(f"prefixo='{prefix}' (~{tipo_guess})")
        if len(digits) >= 8:
            num = digits[:-4].lstrip("0") or "0"
            year = digits[-4:]
            if 1900 <= int(year) <= 2100:
                hints.append(f"número≈{num}, ano≈{year}")
    nums = re.findall(r"\d+", base)
    if nums and "ano≈" not in " ".join(hints):
        for n in reversed(nums):
            if len(n) == 4 and 1900 <= int(n) <= 2100:
                hints.append(f"ano≈{n}")
                break
    return "; ".join(hints)


def _build_prompt(text: str, filename: str, parent_dir: str, has_revocation_hints: bool) -> str:
    revocation_note = (
        "\n\nATENÇÃO: Foram detectados possíveis indicadores de revogação neste documento. "
        "Verifique cuidadosamente se o ato realmente foi revogado ou apenas menciona revogação de outros atos."
        if has_revocation_hints else ""
    )
    hint_line = _filename_hint(filename, parent_dir)
    return f"""Analise o ato normativo abaixo e extraia os metadados em formato JSON.

INSTRUÇÕES IMPORTANTES:
1. STATUS DE VIGÊNCIA:
   - "REVOGADO" se houver menção explícita de revogação do PRÓPRIO ato (texto riscado/tachado, "revogado", "sem efeito", "revoga-se este ato").
   - "VIGENTE" caso contrário (incluindo casos em que o ato apenas revoga OUTROS atos).

2. TIPO: identifique exatamente (Resolução, Portaria, Portaria Conjunta, Provimento, Aviso, Aviso Conjunto, Instrução, Instrução de Serviço, Ordem de Serviço, Emenda Regimental, Deliberação, Enunciado, Regimento Interno, etc.). Se o documento for um artigo/trecho isolado de um Regimento Interno, use tipo="Regimento Interno".

3. NÚMERO e ANO: extraia do cabeçalho do documento. Se o cabeçalho NÃO contiver, use as DICAS do nome do arquivo abaixo. Só use "0" e 0 se for impossível inferir.

4. ÓRGÃO: identifique o órgão emissor (Presidência, Corregedoria, 1ª Vice-Presidência, 2ª Vice-Presidência, ASPREC, NUIREF, etc.).

5. ASSUNTO: resumo conciso (1 frase, no máximo 200 chars) do tema principal.

6. TAGS: 3-5 palavras-chave relevantes para busca.{revocation_note}

ARQUIVO: {filename}
DICAS DO ARQUIVO: {hint_line}

Responda APENAS com o JSON, sem markdown ou explicações:
{{
  "tipo": "string",
  "numero": "string",
  "ano": int,
  "orgao": "string",
  "status": "VIGENTE" | "REVOGADO",
  "assunto_resumo": "string",
  "tags": ["lista", "de", "tags"]
}}

TEXTO DO ATO NORMATIVO:
{text[:50000]}
"""


def classify_with_llm(text: str, filename: str = "", parent_dir: str = "") -> Optional[Dict[str, Any]]:
    """Use Azure OpenAI to classify a normative act document."""
    client = get_azure_client()
    if client is None:
        return None

    has_revocation_hints = detect_strikethrough_patterns(text)
    prompt = _build_prompt(text, filename, parent_dir, has_revocation_hints)

    try:
        response = client.chat.completions.create(
            model=AZURE_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2048,
            response_format={"type": "json_object"},
        )
        response_text = (response.choices[0].message.content or "").strip()
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        metadata = json.loads(response_text)

        for field in ("tipo", "numero", "ano", "status"):
            if field not in metadata:
                logger.warning(f"Missing field '{field}' in LLM response for {filename}")
                return None

        status = str(metadata.get("status", "")).upper()
        metadata["status"] = status if status in ("VIGENTE", "REVOGADO") else "VIGENTE"

        try:
            metadata["ano"] = int(metadata.get("ano") or 0)
        except (TypeError, ValueError):
            metadata["ano"] = 0

        metadata["numero"] = str(metadata.get("numero") or "0")
        metadata.setdefault("orgao", None)
        metadata.setdefault("assunto_resumo", "")
        metadata.setdefault("tags", [])

        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON for {filename}: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM classification error for {filename}: {e}")
        return None


def _split_oversized(piece: str, max_size: int, overlap: int = 200) -> list:
    """Split a single oversized piece into chunks of <= max_size with overlap."""
    if len(piece) <= max_size:
        return [piece]
    out = []
    step = max(1, max_size - overlap)
    for start in range(0, len(piece), step):
        sub = piece[start:start + max_size].strip()
        if sub:
            out.append(sub)
        if start + max_size >= len(piece):
            break
    return out


def chunk_by_articles(text: str, metadata: Dict[str, Any], max_chunk_size: int = 1500) -> list:
    """Chunk text semantically by legal articles, with metadata context prefix.

    Guarantees that no resulting chunk exceeds ``max_chunk_size`` characters
    (oversized articles get split with overlap).
    """
    context_prefix = (
        f"[{metadata.get('tipo', 'Ato')} {metadata.get('numero', '')}/"
        f"{metadata.get('ano', '')} - {metadata.get('status', 'VIGENTE')}] "
    )
    article_pattern = r'(?=\n\s*Art\.?\s*\d+|(?<=\n)\s*§\s*\d+|(?<=\n)\s*Parágrafo\s+único)'
    raw_chunks = [c.strip() for c in re.split(article_pattern, text) if c.strip()]

    merged: list = []
    current = ""
    for chunk in raw_chunks:
        if len(chunk) > max_chunk_size:
            if current:
                merged.append(current.strip())
                current = ""
            merged.extend(_split_oversized(chunk, max_chunk_size))
            continue
        if len(current) + len(chunk) < max_chunk_size:
            current = (current + "\n" + chunk) if current else chunk
        else:
            if current:
                merged.append(current.strip())
            current = chunk
    if current:
        merged.append(current.strip())

    if len(merged) <= 1 and len(text) > max_chunk_size:
        merged = _split_oversized(text, max_chunk_size)

    final = [f"{context_prefix}{c}" for c in merged if c]
    return final or [f"{context_prefix}{text[:max_chunk_size]}"]
