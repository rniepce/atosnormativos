"""
LLM-based classifier for normative acts using Gemini 3 Flash.
Detects document type, status (VIGENTE/REVOGADO), and extracts metadata.
"""
import os
import json
import logging
import re
from typing import Dict, Any, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Lazy initialization of Gemini model
_GEMINI_MODEL = None


def get_classifier_model():
    """Initialize Gemini model for classification."""
    global _GEMINI_MODEL
    if _GEMINI_MODEL is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            # Try Gemini 3 Flash Preview first, then fallbacks
            model_names = ['gemini-3-flash-preview', 'gemini-2.0-flash', 'gemini-1.5-flash']
            for model_name in model_names:
                try:
                    _GEMINI_MODEL = genai.GenerativeModel(model_name)
                    logger.info(f"Classifier using model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Model {model_name} not available: {e}")
            if _GEMINI_MODEL is None:
                logger.error("No Gemini model could be loaded for classification")
        else:
            logger.warning("GEMINI_API_KEY not set, LLM classification disabled")
    return _GEMINI_MODEL


def detect_strikethrough_patterns(text: str) -> bool:
    """
    Detect patterns that indicate revoked text:
    - Unicode strikethrough characters
    - Common revocation phrases
    - RTF/Word strikethrough markers
    """
    # Common revocation phrases in Brazilian legal documents
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
    
    # Check for Unicode strikethrough characters (U+0336)
    if '\u0336' in text:
        return True
    
    return False


def classify_with_llm(text: str, filename: str = "") -> Optional[Dict[str, Any]]:
    """
    Use Gemini to classify a normative act document.
    
    Args:
        text: Full text content of the document
        filename: Original filename for hints
        
    Returns:
        Dictionary with classification metadata or None if failed
    """
    model = get_classifier_model()
    
    if model is None:
        logger.warning("LLM classification not available, using fallback")
        return None
    
    # Pre-check for obvious revocation patterns
    has_revocation_hints = detect_strikethrough_patterns(text)
    
    prompt = f"""Você é um classificador especializado em atos normativos do Tribunal de Justiça de Minas Gerais (TJMG).

Analise o ato normativo abaixo e extraia os metadados em formato JSON.

INSTRUÇÕES IMPORTANTES:
1. **STATUS DE VIGÊNCIA**: 
   - Marque como "REVOGADO" se:
     * O texto contiver menção explícita de revogação (ex: "revogada", "sem efeito", "revoga-se")
     * O texto estiver riscado ou tachado
     * Houver indicação de que a norma perdeu eficácia
   - Caso contrário, marque como "VIGENTE"

2. **TIPO**: Identifique o tipo exato (Resolução, Portaria, Portaria Conjunta, Provimento, Aviso, Instrução, Ordem de Serviço, Emenda Regimental, etc.)

3. **NÚMERO e ANO**: Extraia com precisão do cabeçalho do documento

4. **ÓRGÃO**: Identifique o órgão emissor (Presidência, Corregedoria, Vice-Presidência, etc.)

5. **ASSUNTO**: Faça um resumo conciso do tema principal

6. **TAGS**: Liste 3-5 palavras-chave relevantes para busca

{"ATENÇÃO: Foram detectados possíveis indicadores de revogação neste documento. Verifique cuidadosamente." if has_revocation_hints else ""}

ARQUIVO: {filename}

Responda APENAS com o JSON, sem markdown ou explicações:
{{
  "tipo": "string",
  "numero": "string",
  "ano": int,
  "orgao": "string",
  "status": "VIGENTE" ou "REVOGADO",
  "assunto_resumo": "string",
  "tags": ["lista", "de", "tags"]
}}

TEXTO DO ATO NORMATIVO:
{text[:50000]}
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.strip()
        # Clean potential markdown code blocks
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        metadata = json.loads(response_text)
        
        # Ensure required fields exist
        required_fields = ["tipo", "numero", "ano", "status"]
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Missing field '{field}' in LLM response")
                return None
        
        # Normalize status
        if metadata.get("status", "").upper() not in ["VIGENTE", "REVOGADO"]:
            metadata["status"] = "VIGENTE"
        else:
            metadata["status"] = metadata["status"].upper()
        
        logger.info(f"LLM classified: {metadata['tipo']} {metadata['numero']}/{metadata['ano']} - {metadata['status']}")
        return metadata
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM classification error: {e}")
        return None


def chunk_by_articles(text: str, metadata: Dict[str, Any], max_chunk_size: int = 1500) -> list:
    """
    Chunk text semantically by legal articles.
    
    Strategy:
    1. Split by article boundaries (Art., §, Inciso)
    2. Merge small consecutive articles
    3. Add metadata context prefix to each chunk
    """
    # Context prefix for each chunk
    context_prefix = f"[{metadata.get('tipo', 'Ato')} {metadata.get('numero', '')}/{metadata.get('ano', '')} - {metadata.get('status', 'VIGENTE')}] "
    
    # Split by article patterns
    article_pattern = r'(?=\n\s*Art\.?\s*\d+|(?<=\n)\s*§\s*\d+|(?<=\n)\s*Parágrafo\s+único)'
    
    raw_chunks = re.split(article_pattern, text)
    
    # Filter empty chunks and strip whitespace
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    
    # Merge small chunks to meet minimum size
    merged_chunks = []
    current_chunk = ""
    
    for chunk in raw_chunks:
        if len(current_chunk) + len(chunk) < max_chunk_size:
            current_chunk += "\n" + chunk if current_chunk else chunk
        else:
            if current_chunk:
                merged_chunks.append(current_chunk.strip())
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk.strip())
    
    # If no article structure found, fall back to simple chunking
    if len(merged_chunks) <= 1 and len(text) > max_chunk_size:
        # Simple chunking with overlap
        merged_chunks = []
        overlap = 200
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            chunk = text[start:end].strip()
            if chunk:
                merged_chunks.append(chunk)
            start = end - overlap
    
    # Add context prefix to each chunk
    final_chunks = [f"{context_prefix}{chunk}" for chunk in merged_chunks if chunk]
    
    return final_chunks if final_chunks else [f"{context_prefix}{text[:max_chunk_size]}"]
