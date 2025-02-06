from pathlib import Path
from typing import Optional, List
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import json

class IndexManager:
    def __init__(self, dataset_dir: str, config):
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.collection_name = config.COLLECTION_NAME

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.config.EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def create_or_load_index(self) -> bool:
        """Create or load Qdrant collection"""
        try:
            collections = self.client.get_collections().collections
            exists = any(collection.name == self.collection_name for collection in collections)
            if not exists:
                print("Creating new Qdrant collection...")
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                if not self.dataset_dir.exists():
                    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
                for signal_file in self.dataset_dir.glob('signal_*.txt'):
                    with open(signal_file, 'r') as f:
                        content = f.read()
                        embedding = self.get_embedding(content)

                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=[PointStruct(
                                id=int(signal_file.stem.split('_')[1]),
                                vector=embedding,
                                payload={"content": content}
                            )]
                        )
                print("Collection created and documents indexed.")
            else:
                print("Using existing Qdrant collection...")

            return True

        except Exception as e:
            print(f"Error creating/loading index: {str(e)}")
            return False

    def rebuild_index(self) -> bool:
        """Rebuild the Qdrant collection"""
        try:
            collections = self.client.get_collections().collections
            if any(collection.name == self.collection_name for collection in collections):
                self.client.delete_collection(collection_name=self.collection_name)

            return self.create_or_load_index()
        except Exception as e:
            print(f"Error rebuilding index: {str(e)}")
            return False

    def search(self, query: str, limit: int = 3) -> List[dict]:
        """Search for similar documents"""
        try:
            query_embedding = self.get_embedding(query)

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["content"],
                    "weight": float(result.score) * 100
                })
            return results
        except Exception as e:
            print(f"Error searching index: {str(e)}")
            return []
