"""
Custom LLM implementation for the LlamaIndex agent workflow.
"""
from llama_index.core.llms import CustomLLM, LLMMetadata
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
API_URL = os.getenv('API_URL', '')
BEARER_TOKEN = os.getenv('BEARER_TOKEN', '')

class APILLM(CustomLLM):
    """Custom LLM implementation using API endpoints."""
    
    context_window: int = 39000
    num_output: int = 2560
    model_name: str = "gpt-4o"
    api_url: str = API_URL
    bearer_token: str = BEARER_TOKEN

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def _make_api_call(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "applicationType": "Developer",
            "Content-Type": "application/json"
        }
        # Add your API call implementation here
        return ""  # Replace with actual implementation
