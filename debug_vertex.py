
import os
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
import vertexai

load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

print(f"GCP_PROJECT_ID: {os.getenv('GCP_PROJECT_ID')}")
print(f"GCP_LOCATION: {os.getenv('GCP_LOCATION')}")
json_creds = os.getenv('GOOGLE_CREDENTIALS_JSON')
print(f"GOOGLE_CREDENTIALS_JSON present: {bool(json_creds)}")

if json_creds:
    try:
        service_account_info = json.loads(json_creds)
        print("JSON decode successful")
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        print("Credentials object created")
        
        vertexai.init(
            project=os.getenv('GCP_PROJECT_ID'), 
            location=os.getenv('GCP_LOCATION'), 
            credentials=credentials
        )
        print("vertexai.init successful")
        
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embeddings = model.get_embeddings(["test"])
        print("Embedding test successful")
        
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No credentials found")
