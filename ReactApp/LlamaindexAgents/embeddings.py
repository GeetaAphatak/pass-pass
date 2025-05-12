"""
Custom embeddings implementation for the LlamaIndex agent workflow.
"""
import logging
import requests
from typing import List
from llama_index.core.embeddings import BaseEmbedding
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)
EMBEDDING_URL = os.getenv('embedding_url', '')
API_KEY = os.getenv('embeddings_bearer_token', '')
EMBEDDING_MODEL_NAME = os.getenv('embedding_model_name', '')

logger = logging.getLogger(__name__)

class CustomEmbeddings(BaseEmbedding):
    """
    Custom embedding class for interacting with a specific embedding API.
    """
    _api_key: str = API_KEY
    _embedding_url: str = EMBEDDING_URL
    _embedding_model_name: str = EMBEDDING_MODEL_NAME
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing CustomEmbeddings class.")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} texts.")
        headers = {
            'applicationType': "Developer",
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._api_key}',
        }
        payload = {
            "model": self._embedding_model_name,
            "input": texts
        }
        try:
            response = requests.post(self._embedding_url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            if 'result' in response_data and 'data' in response_data['result']:
                embeddings = [item['embedding'] for item in response_data['result']['data']]
                logger.info(f"Received {len(embeddings)} embeddings successfully.")
                return embeddings
            else:
                raise ValueError(f"Unexpected response format: {response_data}")
        except requests.RequestException as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]
        
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    @property
    def embed_dim(self) -> int:
        return 1536

    @property
    def query_embed_dim(self) -> int:
        return self.embed_dim
