from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.vertex import VertexAIClient

class LegalTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Specific separators for Brazilian legal text
        separators = [
            "\nCAPÍTULO", 
            "\nSeção", 
            "\nArt.", 
            "\nParágrafo", 
            "\n§", 
            "\n\n", 
            "\n", 
            " "
        ]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True
        )
        self.vertex_client = VertexAIClient()

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits text into chunks and enriches them with metadata context.
        Returns a list of dictionaries with 'content' and 'embedding'.
        """
        raw_chunks = self.splitter.split_text(text)
        
        # Context enrichment string
        # Ex: "Norma: Portaria 123/2023 (Vigente) > "
        context_prefix = f"Norma: {metadata.get('tipo', 'Norma')} {metadata.get('numero', '')}/{metadata.get('ano', '')} ({metadata.get('status', 'Indefinido')}) > Conteúdo: "
        
        enriched_chunks_data = []
        texts_to_embed = []
        
        for chunk in raw_chunks:
            enriched_text = f"{context_prefix}{chunk}"
            texts_to_embed.append(enriched_text)
            enriched_chunks_data.append({
                "conteudo_texto": chunk, # Store original chunk text for display
                "enriched_text": enriched_text # Use this for embedding if needed, or just strict embedding
            })
            
        # Optimization: Batch embed
        # Vertex AI supports batch embedding (up to 250 inputs usually)
        embeddings = self.vertex_client.get_embeddings(texts_to_embed)
        
        final_chunks = []
        for i, data in enumerate(enriched_chunks_data):
            final_chunks.append({
                "conteudo_texto": data["conteudo_texto"],
                "embedding": embeddings[i]
            })
            
        return final_chunks
