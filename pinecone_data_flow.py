import os
import json
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from prefect import flow, task

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
# from pinecone.core.exceptions import NotFoundException

# from langchain_text_splitters.character import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import HTMLHeaderTextSplitter


@task
def start(config):
    """
    Start-up: check everything works or fail fast!
    """

    # Print out some debug info
    print("Starting flow!")

    # Loading environment variables
    try:
        load_dotenv(verbose=True, dotenv_path='.env')
    except ImportError:
        print("Env file not found!")

    # Ensure user has set the appropriate env variables
    assert os.environ['GOOGLE_API_KEY']
    assert os.environ['PINECONE_API_KEY']
    assert os.environ['PINECONE_INDEX_NAME']

    EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
    EMBEDDING_DIMENSION = config["EMBEDDING_DIMENSION"]



"""Get Data"""

PATCH_LISST_URL = "https://www.leagueoflegends.com/en-us/news/tags/patch-notes/"
JSON_FILE_PATH = os.path.join(os.getcwd(), "data", "patch_data.json")

@task
def get_latest_patch_version():
    response = requests.get(PATCH_LISST_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Get first (latest) patch
        latest_patch_element = soup.find("a", {"data-testid": "articlefeaturedcard-component"})
        if latest_patch_element:
            aria_label = latest_patch_element.get("aria-label", "")
            if "Patch" in aria_label:
                 # return version id, example: "14.22", and url
                return aria_label.split(" ")[1], latest_patch_element.get("href", "") 

    print("Error: Unable to retrieve the latest patch version.")  
    return None, None

@task
def is_newer_patch(latest_patch_version):
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r") as file:
            patch_data = json.load(file)
    else:
        print("json data not found, create new data file.")
        return True

    current_version = patch_data.get("version", "")
    if current_version == "":
        print("Error: Unable to retrieve the current patch version (Version id not found).")
        return False
    
    if current_version != latest_patch_version:
        return True
    else:
        return False

@task  
def update_patch_info(soup, version, url):
    title = soup.find("h1", {"data-testid": "title"}).text.strip()

    # Lấy mô tả (description)
    description = soup.find("div", {"data-testid": "tagline"}).text.strip()

    # Lấy thời gian (time)
    time = soup.find("time").text.strip()

    # Lấy ảnh tổng quan của patch (overview image)
    patch_highlights_header = soup.find("h2", {"id": "patch-patch-highlights"})
    overview_image_url = None
    if patch_highlights_header:
        # Tìm ảnh đầu tiên sau h2 patch highlights
        overview_image = patch_highlights_header.find_next("img")
        if overview_image and "src" in overview_image.attrs:
            overview_image_url = overview_image["src"]

    # Tạo dữ liệu dưới dạng dictionary
    patch_data = {
        "version": version,
        "title": title,
        "description": description,
        "time": time,
        "url": url,
        "overview_image": overview_image_url
    }

    # Lưu vào file JSON
    with open(JSON_FILE_PATH, "w") as json_file:
        json.dump(patch_data, json_file, ensure_ascii=False, indent=4)

    print("Dữ liệu mới đã được lưu vào file patch_data.json")

    return soup

@task
def get_html_patch(soup):
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

    return full_html_content


"""Parse data to docs"""

@task
def convert_html_to_docs(html_content):
    headers_to_split_on = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
        ("h4", "h4"),
        ("h5", "h5"),
    ]

    splitter = HTMLHeaderTextSplitter(headers_to_split_on)

    # Tách nội dung thành các đoạn theo các thẻ tiêu đề
    docs = splitter.split_text(html_content)

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

    return docs


"""Load data to pinecone"""

@task
def upload_docs_to_pinecone(docs, config):
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=config['EMBEDDING_DIMENSION'],
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ))

    # Target index and check status
    pc_index = pc.Index(PINECONE_INDEX_NAME)
    print(pc_index.describe_index_stats())

    embeddings = GoogleGenerativeAIEmbeddings(model=config['EMBEDDING_MODEL_NAME'])
    namespace = "lol-patch"

    try:
        pc_index.delete(namespace=namespace, delete_all=True)
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


@flow(log_prints=True)
def pinecone_flow():
    with open('./config.json') as f:
        config = json.load(f)
    
    print(config)
    
    start(config)
    latest_patch_version, url = get_latest_patch_version()
    full_url = f"https://www.leagueoflegends.com{url}"
    if is_newer_patch(latest_patch_version):
        print("New patch found. Updating ...")

        response = requests.get(f"{full_url}")
        if response.status_code == 200:
            page_content = response.content
        else:
            print(f"[404] - URL {full_url} not found")
            return
        soup = BeautifulSoup(page_content, "html.parser")

        html = get_html_patch(soup)
        docs = convert_html_to_docs(html)
        upload_docs_to_pinecone(docs, config)

        update_patch_info(soup, latest_patch_version, full_url)
    else:
        print("The current patch is the latest")
