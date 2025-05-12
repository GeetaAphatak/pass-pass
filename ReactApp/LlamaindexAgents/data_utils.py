"""
Data processing utilities for the LlamaIndex agent workflow.
"""
import json
import pandas as pd
from typing import List, Dict, Any
from llama_index.core.schema import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from .llm import APILLM

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def create_documents_from_json(apps_data: List[Dict[str, Any]]) -> List[Document]:
    """Create Document objects from JSON data."""
    documents = []
    for app in apps_data:
        doc = Document(text=json.dumps(app))
        documents.append(doc)
    return documents

def create_rag_system(
    documents: List[Document], 
    custom_embeddings: BaseEmbedding, 
    llm: APILLM
) -> RetrieverQueryEngine:
    """Create RAG system from documents."""
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    
    index = VectorStoreIndex(
        nodes,
        embed_model=custom_embeddings,
        llm=llm,
    )
    
    return index.as_query_engine()

def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """Load CSV file into pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return pd.DataFrame()

def query_csv_dataframe(df: pd.DataFrame, query: str) -> str:
    """Query the CSV DataFrame."""
    try:
        # Add your DataFrame query implementation here
        return str(df.head())  # Replace with actual query implementation
    except Exception as e:
        return f"Error querying DataFrame: {str(e)}"

def generate_csv_description(llm: APILLM, csv_df: pd.DataFrame) -> str:
    """Generate description for CSV data."""
    # Add your CSV description generation implementation here
    return "CSV Description"  # Replace with actual implementation
