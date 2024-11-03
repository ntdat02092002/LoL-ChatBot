import os
import json
from dotenv import load_dotenv
from typing import Optional
from typing import Dict

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import StructuredQuery
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
)

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from prompt_template import get_chatbot_prompt, get_constructor_prompt


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
        self.constructor_prompt = get_constructor_prompt(type="custom")

    def initialize_vector_store(self):
        # Create empty index
        PINECONE_KEY, PINECONE_INDEX_NAME = os.getenv(
            'PINECONE_API_KEY'), os.getenv('PINECONE_INDEX_NAME')

        pc = Pinecone(api_key=PINECONE_KEY)

        # Target index and check status
        pc_index = pc.Index(PINECONE_INDEX_NAME)

        embeddings = HuggingFaceEmbeddings()

        namespace = "lol-patch"
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

        prompt = get_chatbot_promp()

        self.rag_chain_with_source = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            # | RunnableLambda(self.log_context_and_question) 
            | prompt 
            | chat_model 
            | StrOutputParser()
        )