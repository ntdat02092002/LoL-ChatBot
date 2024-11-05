import os
import json
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
# from pinecone.core.exceptions import NotFoundException

# from langchain_text_splitters.character import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import HTMLHeaderTextSplitter


load_dotenv()
with open('./config.json') as f:
    config = json.load(f)
    EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]

"""Get Data"""

# Lấy nội dung trang HTML
url = "https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-21-notes/"
response = requests.get(url)
if response.status_code == 200:
    page_content = response.content
else:
    raise f"[404] - URL not found - {url}"

# Parse HTML và lấy phần có id="patch notes"
soup = BeautifulSoup(page_content, "html.parser")
patch_section = soup.find(id="patch-notes-container")

# Danh sách các thẻ chỉ có tác dụng định dạng mà không ảnh hưởng tới nội dung
formatting_tags = ["strong", "em", "u", "b", "i", "s"]

# Nối tất cả nội dung của thẻ và thẻ con vào một chuỗi, giữ nguyên HTML
full_html_content = ""
if patch_section:
    for child in patch_section.find_all(recursive=False):
        # Chèn \n sau phần văn bản nếu có nội dung thực sự và không thuộc thẻ định dạng
        for elem in child.find_all(text=True):
            parent_tag = elem.parent.name
            if elem.strip():  # Kiểm tra chuỗi có nội dung thực sự
                # Chỉ chèn \n nếu thẻ cha không nằm trong danh sách các thẻ định dạng
                if parent_tag not in formatting_tags:
                    elem.replace_with(elem + "\\n")
        full_html_content += str(child)  # Giữ nguyên HTML của các thẻ con
else:
    raise "cannot find <id='patch notes'> element"


"""Parse data to docs"""

headers_to_split_on = [
    ("h1", "h1"),
    ("h2", "h2"),
    ("h3", "h3"),
    ("h4", "h4"),
    ("h5", "h5"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on)

# Tách nội dung thành các đoạn theo các thẻ tiêu đề
docs = splitter.split_text(full_html_content)

for doc in docs:
    # first doc - overview patch with no header (no metadata)
    if not doc.metadata:
        doc.metadata = {"category": "overview"}
        continue

    page_content = ""
    type = "other"
    for key, value in doc.metadata.items():
        value = value.strip("\\n").strip("\n") # strip \n - the addition when process by beautifulsoup
        page_content += f"<{key}>{value}<{key}>\n"
        if key == "h2" and value == "Champions":
            type = "champion"
        elif key == "h2" and value == "Items":
            type = "item"
            
    page_content += doc.page_content

    doc.page_content = page_content
    doc.metadata = {"category": type}


"""Load data to pinecone"""

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

PINECONE_INDEX_NAME = "docs-rag-chatbot"
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))

# Target index and check status
pc_index = pc.Index(PINECONE_INDEX_NAME)
print(pc_index.describe_index_stats())

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
namespace = "lol-patch"

try:
    pc_index.delete(namespace=namespace, delete_all=True)
# except NotFoundException:
#     print(f"Namespace '{namespace}' not found. Not deleting.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
else:
    print("Namespace deleted successfully.")

PineconeVectorStore.from_documents(
    docs,
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=namespace
)

print("Successfully uploaded docs to Pinecone vector store")
