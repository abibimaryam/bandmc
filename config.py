from langchain_community.llms.ollama import Ollama

DATA_DIR = "/home/maryam/Документы/projects/bandmc/data"
CHROMA_PATH = "chroma_db"
model = Ollama(model="gemma3:4b", temperature=0)