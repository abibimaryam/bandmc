from langchain_community.llms.ollama import Ollama
# from llama_index.llms.ollama import Ollama

DATA_DIR = "/home/maryam/Документы/projects/bandmc/data"
CHROMA_PATH = "chroma_db"
# model = Ollama(model="medllama2:latest", temperature=0)
model=Ollama(model="alibayram/medgemma:latest")
# model=Ollama(model="tripolskypetr/gemma3-tools:4b", temperature=0,request_timeout=180.0)

