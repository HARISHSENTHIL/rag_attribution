import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    DATASET_DIR = os.getenv("DATASET_DIR")

    @classmethod
    def setup(cls):
        """Setup OpenAI client"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return OpenAI(api_key=cls.OPENAI_API_KEY)
