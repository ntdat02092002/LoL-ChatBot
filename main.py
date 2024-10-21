import os
import json
from dotenv import load_dotenv
from typing import Optional
from typing import Dict

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo, StructuredQuery
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


class LolChatBot():
    RETRIEVER_MODEL_NAME: str = None
    SUMMARY_MODEL_NAME: str = None
    EMBEDDING_MODEL_NAME: str = None
    constructor_prompt: Optional[PromptTemplate] = None
    vectorstore: Optional[PineconeVectorStore] = None
    retriever: Optional[SelfQueryRetriever] = None
    rag_chain_with_source: Optional[RunnableParallel] = None
    query_constructor: RunnableSerializable[Dict, StructuredQuery] = None
    context: str = None
    top_k: int = None

    def __init__(self, **kwargs):
        print("------------------init bot------------------")
        load_dotenv()
        with open('./config.json') as f:
            config = json.load(f)
            self.RETRIEVER_MODEL_NAME = config["RETRIEVER_MODEL_NAME"]
            self.SUMMARY_MODEL_NAME = config["SUMMARY_MODEL_NAME"]
            self.EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
            self.top_k = config["top_k"]
        self.initialize_query_constructor()
        self.initialize_vector_store()
        self.initialize_retriever()
        self.initialize_chat_model(config)

    def initialize_query_constructor(self):
        document_content_description = "Brief overview of the game LoL update info"

        # Define allowed comparators list
        allowed_comparators = [
            "$eq",  # Equal to (number, string, boolean)
            "$ne",  # Not equal to (number, string, boolean)
            "$gt",  # Greater than (number)
            "$gte",  # Greater than or equal to (number)
            "$lt",  # Less than (number)
            "$lte",  # Less than or equal to (number)
            "$in",  # In array (string or number)
            "$nin",  # Not in array (string or number)
        ]

        # Define allowed operators list
        allowed_operators = [
            "AND",
            "OR"
        ]

        examples = [
            (
                "What changes does Zoe have?",
                {
                    "query": "Zoe",
                    "filter": "eq(\"type\", \"champion\")"
                }
            ),
            (
                "Was Statikk Shiv buffed or nerfed?",
                {
                    "query": "Statikk Shiv",
                    "filter": "eq(\"type\", \"item\")"
                }
            ),
            (
                "What is the latest version?",
                {
                    "query": "latest version",
                    "filter": "NO_FILTER"
                }
            ),
            (
                "Which champions were buffed in the latest patch?",
                {
                    "query": "buff",
                    "filter": "eq(\"type\", \"champion\")"
                }
            ),
            (
                "Were any items nerfed in the latest patch?",
                {
                    "query": "nerf",
                    "filter": "eq(\"type\", \"item\")"
                }
            ),
            (
                "Are there any buffs for Jinx or Infinity Edge?",
                {
                    "query": "buff Jinx Infinity Edge",
                    "filter": "or(eq(\"type\", \"champion\"), eq(\"type\", \"item\"))"
                }
            ),
        ]

        metadata_field_info = [
            AttributeInfo(name="type", description="The type of the update. One of ['champion', 'item']",
                          type="string"),
        ]

        self.constructor_prompt = get_query_constructor_prompt(
            document_content_description,
            metadata_field_info,
            allowed_comparators=allowed_comparators,
            allowed_operators=allowed_operators,
            examples=examples,
        )

    def initialize_vector_store(self):
        # Create empty index
        PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
            'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')

        pc = Pinecone(api_key=PINECONE_KEY)

        # Target index and check status
        pc_index = pc.Index(PINECONE_INDEX_NAME)

        embeddings = HuggingFaceEmbeddings()

        namespace = "test"
        self.vectorstore = PineconeVectorStore(
            index=pc_index,
            embedding=embeddings,
            namespace=namespace
        )

    def initialize_retriever(self):
        query_model = HuggingFaceEndpoint(
            repo_id=self.RETRIEVER_MODEL_NAME,
            # max_length=128,
            temperature=0.001,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
        )

        output_parser = StructuredQueryOutputParser.from_components()
        self.query_constructor = self.constructor_prompt | query_model | output_parser

        self.retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.vectorstore,
            structured_query_translator=PineconeTranslator(),
            search_kwargs={'k': self.top_k}
        )

    def log_context_and_question(inputs):
        context = inputs["context"]
        question = inputs["question"]
        
        # Print to console (or you can use logging for better debugging)
        print(f"Context retrieved: {context}")
        print(f"User question: {question}")
        print("-------------------------------------------------------------------------")
        
        # Return the inputs so they can be passed forward in the chain
        return inputs

    def initialize_chat_model(self, config):

        chat_model = HuggingFaceEndpoint(
            repo_id=self.RETRIEVER_MODEL_NAME,
            # max_length=128,
            temperature=config['TEMPERATURE'],
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
        )

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             'system',
        #             """
        #             Your goal is to recommend films to users based on their
        #             query and the retrieved context. If a retrieved film doesn't seem
        #             relevant, omit it from your response. If your context is empty
        #             or none of the retrieved films are relevant, do not recommend films, but instead
        #             tell the user you couldn't find any films that match their query.
        #             Aim for three to five film recommendations, as long as the films are relevant. You cannot
        #             recommend more than five films. Your recommendation should
        #             be relevant, original, and at least two to three sentences
        #             long.

        #             YOU CANNOT RECOMMEND A FILM IF IT DOES NOT APPEAR IN YOUR
        #             CONTEXT.

        #             # TEMPLATE FOR OUTPUT
        #             - **Title of Film**:
        #                 - **Runtime:**
        #                 - **Release Year:**
        #                 - **Streaming:**
        #                 - Your reasoning for recommending this film

        #             Question: {question}
        #             Context: {context}
        #             """
        #         ),
        #     ]
        # )

        template = """
        You are a League of Legends expert. Players will ask you questions about patch updates. 
        Use the following context to answer the question. This is the information support you answer the user's questions.
        If you don't know the answer, just say you don't know. 
        Keep the answer relevant to the patch update and concise.


        Context: {context}
        Question: {question}
        Answer: 

        """

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # Create a chatbot Question & Answer chain from the retriever
        # rag_chain_from_docs = (
        #     RunnableLambda(self.log_context_and_question) | prompt | chat_model | StrOutputParser()
        # )

        # self.rag_chain_with_source = RunnableParallel(
        #     {"context": self.retriever, "question": RunnablePassthrough(), "query_constructor": self.query_constructor}
        # ).assign(answer=rag_chain_from_docs)
        self.rag_chain_with_source = (
            {"context": self.retriever, "question": RunnablePassthrough(), "query_constructor": self.query_constructor}
            # | RunnableLambda(self.log_context_and_question) 
            | prompt 
            | chat_model 
            | StrOutputParser()
        )