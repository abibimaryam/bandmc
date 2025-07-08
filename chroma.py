import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from embedding_functiom import embedding_function

# Пути к файлам и базе
DATA_PATH = "/home/maryam/Документы/projects/bandmc/data/1.xlsx"
CHROMA_PATH = "chroma_db"

# Загружаем Excel и очищаем пропуски
df = pd.read_excel(DATA_PATH, skiprows=1)
df = df.fillna("")

# Нормализуем названия колонок: убираем переносы строк и лишние пробелы
df.columns = [' '.join(col.strip().split()) for col in df.columns]

# print("Очищенные названия колонок:")
# print(df.columns.tolist())

# Инициализируем эмбеддинг функцию
embedding_func = embedding_function()

# Создаем подключение к ChromaDB
vectordb = Chroma(
    collection_name="meds",
    embedding_function=embedding_func,
    persist_directory=CHROMA_PATH,
)

# Формируем документы из строк таблицы
docs = []
for idx, row in df.iterrows():
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    metadata = {
        "торговое_название": row["Торговое название и синоним"],
        "международное_название": row["Международное название"]
    }
    doc = Document(page_content=content, metadata=metadata)
    docs.append(doc)

print(f"Всего документов для добавления: {len(docs)}")

# Добавляем документы в базу и сохраняем
vectordb.add_documents(docs)
vectordb.persist()

print(f"✅ Загружено {len(docs)} документов в ChromaDB.")

print(f"Документов в базе после добавления: {vectordb._collection.count()}")
