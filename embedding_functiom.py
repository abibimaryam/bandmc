from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings



def embedding_function():
    # embeddings = BedrockEmbeddings(
        #     credentials_profile_name="default", region_name="us-east-1"
        # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = SentenceTransformerEmbeddings(model_name="nps798/drug-appearance-similarity-embedding-embedding-thenlper-gte-small")
    return embeddings
