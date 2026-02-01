import os
from typing import Optional, List
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Initialize Vertex AI
# Ideally initialized via environment variables or explicitly passed
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "southamerica-east1")

if PROJECT_ID:
    vertexai.init(project=PROJECT_ID, location=LOCATION)

class VertexAIClient:
    def __init__(self, project_id: str = PROJECT_ID, location: str = LOCATION):
        if not project_id:
             # Fallback or error if not set, though init usually handles it if gcloud is auth'd
             pass
        self.project_id = project_id
        self.location = location
        self._embedding_model = None

    def get_embedding_model(self):
        if not self._embedding_model:
            self._embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        return self._embedding_model

    def generate_text(self, 
                      prompt: str, 
                      model_name: str = "gemini-2.0-flash-exp",
                      temperature: float = 0.0,
                      response_mime_type: Optional[str] = None) -> str:
        
        model = GenerativeModel(model_name)
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 8192,
        }
        if response_mime_type:
             generation_config["response_mime_type"] = response_mime_type

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        )
        return response.text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        model = self.get_embedding_model()
        inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
        embeddings = model.get_embeddings(inputs)
        return [embedding.values for embedding in embeddings]

    def get_query_embedding(self, text: str) -> List[float]:
        model = self.get_embedding_model()
        inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")]
        embeddings = model.get_embeddings(inputs)
        return embeddings[0].values
