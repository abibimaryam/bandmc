# from llama_index.core.agent.workflow import AgentWorkflow
# from config import model,CHROMA_PATH
# from langchain_community.vectorstores import Chroma
# from embedding_functiom import embedding_function
# import asyncio

# embedding_func = embedding_function()

# vectordb = Chroma(
#         collection_name="meds",
#         embedding_function=embedding_func,
#         persist_directory=CHROMA_PATH,
#     )

# # Сделаем функцию поиска по базе Chroma:
# def search_tool_func(query: str) -> str:
#     # Сделаем поиск, вернём конкатенацию текстов документов
#     results = vectordb._collection.query(
#         query_texts=[query],
#         n_results=5,
#         include=['documents']
#     )
#     docs = results['documents'][0]
#     print(docs)  # список строк документов
#     return "\n\n".join(docs)


# # Теперь создаём агента с твоим llm и search_tool
# generate_agent = AgentWorkflow.from_tools_or_functions(
#     tools_or_functions=[search_tool_func],
#     llm=model,
#     system_prompt=(
#         "Ты интеллектуальный помощник. "
#         "Для ответа используй сначала поиск по базе данных с помощью инструмента chroma_search, "
#         "затем сформируй полный ответ."
#     )
# )



# async def main():
#     response = await generate_agent.run("Расскажи про Азимакс")
#     print(response)

# asyncio.run(main())


# from llama_index.core.agent.workflow import AgentWorkflow
# from llama_index.core.tools import FunctionTool
# from config import model, CHROMA_PATH
# from langchain_community.vectorstores import Chroma
# from embedding_functiom import embedding_function
# import asyncio

# # embedding_func = embedding_function()

# # vectordb = Chroma(
# #     collection_name="meds",
# #     embedding_function=embedding_func,
# #     persist_directory=CHROMA_PATH,
# # )

# # def search_tool_func(query: str) -> str:
# #     results = vectordb._collection.query(
# #         query_texts=[query],
# #         n_results=5,
# #         include=['documents']
# #     )
# #     docs = results['documents'][0]
# #     print(docs)
# #     return "\n\n".join(docs)

# # # Оборачиваем функцию в Tool
# # search_tool = FunctionTool.from_defaults(search_tool_func)

# # generate_agent = AgentWorkflow.from_tools_or_functions(
# #     tools_or_functions=[search_tool],  # передаём Tool, а не функцию напрямую
# #     llm=model,
# #     system_prompt=(
# #         "Ты интеллектуальный помощник. "
# #         "Для ответа используй сначала поиск по базе данных с помощью инструмента chroma_search, "
# #         "затем сформируй полный ответ."
# #     )
# # )

# # async def main():
# #     response = await generate_agent.run("Расскажи про Азимакс")
# #     print(response)

# # asyncio.run(main())
