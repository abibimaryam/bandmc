from langchain_community.llms.ollama import Ollama


DATA_DIR = "/home/maryam/Документы/projects/bandmc/data"
CHROMA_PATH = "chroma_db"

# model=Ollama(model="alibayram/medgemma:latest",temperature=0)


from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key="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",
    model="gpt-4o-mini",  
    temperature=0
)