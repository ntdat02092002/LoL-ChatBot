import os
from prefect import flow
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    flow.from_source(
        source="https://github.com/ntdat02092002/LoL-ChatBot.git",
        entrypoint="pinecone_data_flow.py:pinecone_flow",
    ).deploy(
        name="lolchatbot-data-flow",
        work_pool_name="lolchatbot-pool",
        cron="0 0 * * *",
        job_variables={
            "pip_packages": [
                                "beautifulsoup4==4.12.3", 
                                "python-dotenv==1.0.1", 
                                "pinecone[grpc]==5.3.1", 
                                "python-dotenv==1.0.1", 
                                "langchain-google-genai==2.0.4",
                                "langchain-pinecone==0.2.0",
                                "langchain-text-splitters==0.3.0", 
                                "langchain==0.3.3",
                                "lark==1.2.2"
                            ],
            "env": {
                "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
                "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY"),
                "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME"),
                "PREFECT_API_KEY": os.environ.get("PREFECT_API_KEY"),
            }
        },
        
    )