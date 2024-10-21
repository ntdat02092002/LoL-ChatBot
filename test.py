from main import LolChatBot


chatbot = LolChatBot()


print(chatbot.constructor_prompt.format(query="dummy question"))
# prompt = chatbot.query_constructor.invoke(
#     {
#         "query": "What are some sci-fi movies from the 90's directed by Luc Besson about taxi drivers"
#     }
# )
# print(prompt)

# docs = chatbot.retriever.invoke(
#     "hello"
# )
# print(docs)