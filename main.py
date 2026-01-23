import os
import json
from dotenv import load_dotenv
from typing import Optional
from typing import Dict

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains.query_constructor.base import StructuredQuery
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
)
import langchain

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from prompt_template import get_chatbot_prompt, get_constructor_prompt
from utils import serialize_pydantic_model

# langchain.debug = True

class LolChatBot():
    RETRIEVER_MODEL_NAME: str = None
    CHAT_MODEL_NAME: str = None
    EMBEDDING_MODEL_NAME: str = None
    EMBEDDING_DIMENSION: int = None
    constructor_prompt: Optional[PromptTemplate] = None
    vectorstore: Optional[PineconeVectorStore] = None
    retriever: Optional[SelfQueryRetriever] = None
    rag_chain_with_source: Optional[RunnableParallel] = None
    query_constructor: RunnableSerializable[Dict, StructuredQuery] = None
    context: str = None
    constructed_query: str = None
    top_k: int = None

    def __init__(self, **kwargs):
        print("------------------init bot------------------")
        load_dotenv()
        with open('./config.json') as f:
            config = json.load(f)
            self.RETRIEVER_MODEL_NAME = config["RETRIEVER_MODEL_NAME"]
            self.CHAT_MODEL_NAME = config["CHAT_MODEL_NAME"]
            self.EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
            self.EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]
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

        embeddings = GoogleGenerativeAIEmbeddings(model=self.EMBEDDING_MODEL_NAME, 
                                                  output_dimensionality=self.EMBEDDING_DIMENSION)

        namespace = "lol-patch"
        self.vectorstore = PineconeVectorStore(
            index=pc_index,
            embedding=embeddings,
            namespace=namespace
        )

    def initialize_retriever(self):
        query_model = GoogleGenerativeAI(
            model=self.RETRIEVER_MODEL_NAME,
            temperature=0.0,
            google_api_key=os.getenv('GOOGLE_API_KEY'),
        )

        output_parser = StructuredQueryOutputParser.from_components()
        self.query_constructor = self.constructor_prompt | query_model | output_parser

        self.retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.vectorstore,
            structured_query_translator=PineconeTranslator(),
            search_kwargs={'k': self.top_k}
        )

    def format_docs(self, docs):
        return [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]
    
    def initialize_chat_model(self, config):
        chat_model = GoogleGenerativeAI(
            model=self.CHAT_MODEL_NAME,
            temperature=config['TEMPERATURE'],
            google_api_key=os.getenv('GOOGLE_API_KEY'),
        )

        prompt = get_chatbot_prompt()

        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self.format_docs(x["context"]))) | prompt | chat_model | StrOutputParser()
        )

        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough(), "constructed_query": self.query_constructor}
        ).assign(answer=rag_chain_from_docs)

    def predict_stream(self, query: str):
        try:
            for chunk in self.rag_chain_with_source.stream(query):
                if 'answer' in chunk:
                    yield chunk['answer']
                elif 'context' in chunk:
                    docs = chunk['context']
                    self.context = self.format_docs(docs)
                elif 'constructed_query' in chunk:
                    self.constructed_query = serialize_pydantic_model(chunk['constructed_query'])

        except Exception as e:
            return {'answer': f"An error occurred: {e}"}
    
    def predict(self, query):
        return self.rag_chain_with_source.invoke(query)['answer']