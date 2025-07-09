from langchain_community.llms.ollama import Ollama

DATA_DIR = "/home/maryam/Документы/projects/bandmc/data"
CHROMA_PATH = "chroma_db"
# med_model = Ollama(model="medllama2:latest", temperature=0)
model=Ollama(model="gemma3:4b", temperature=0)

