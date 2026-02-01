import json
import logging
from typing import Dict, Any, Optional
from src.utils.vertex import VertexAIClient

logger = logging.getLogger(__name__)

class DocumentClassifier:
    def __init__(self, vertex_client: VertexAIClient):
        self.vertex_client = vertex_client

    def classify_document(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses Gemini 1.5 Flash to classify the document and extract metadata.
        """
        prompt = """
        Você é um classificador jurídico especializado em atos normativos do TJMG.
        Analise o texto do ato normativo fornecido abaixo.
        
        Sua tarefa é extrair os metadados em formato JSON, seguindo estritamente este schema:
        {
          "tipo": "Portaria|Resolução|Provimento|Recomendação|Instrução Normativa|Outro",
          "numero": "string (ex: 1234)",
          "ano": int (ex: 2023),
          "orgao": "string (ex: Corregedoria, Presidência)",
          "status": "VIGENTE|REVOGADO",
          "assunto_resumo": "Resumo conciso do que trata a norma",
          "tags": ["lista", "de", "tags", "relevantes"]
        }

        Instruções adicionais:
        1. Identifique se a norma está revogada procurando por termos como 'revoga-se', 'perde efeito' ou menções expressas de revogação no próprio texto (embora geralmente a revogação venha de normas posteriores, tente inferir pelo texto se for uma republicação ou se houver indicativos). Se não houver indicativo claro de revogação, assuma VIGENTE.
        2. Extraia o número e ano com precisão.
        3. Gere tags relevantes para busca (ex: "RH", "Plantão", "Licitação").
        
        Texto do Ato Normativo:
        """
        
        # Truncate text if too long (though 1.5 Flash has 1M context, best to be safe/efficient)
        # 100k chars is plenty to get the header and context
        truncated_text = text[:100000] 
        
        full_prompt = f"{prompt}\n\n{truncated_text}"

        try:
            response_text = self.vertex_client.generate_text(
                prompt=full_prompt,
                model_name="gemini-2.0-flash-exp",
                temperature=0.0,
                response_mime_type="application/json"
            )
            
            # Clean up potential markdown code blocks if any (though mime_type json usually prevents this)
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            metadata = json.loads(response_text)
            return metadata

        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from Gemini response.")
            logger.debug(f"Raw response: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return None
