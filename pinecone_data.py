import os
from dotenv import load_dotenv

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
# from pinecone.core.exceptions import NotFoundException

# from langchain_text_splitters.character import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore


"""Test Data"""

docs = [
    Document(
        page_content="Elise: Passive spiderlings now always regen when resummoned and jump on Q cast. Human E now reveals stealthed units and can now be redirected via flash. Spider E buff now applies on cast and can be recast to immediately drop back. R cooldown decreased.",
        metadata={"type": "champion"},
    ),
    Document(
        page_content="K'Sante: Base Attack Range reduced. All abilities adjusted.",
        metadata={"type": "champion"},
    ),
    Document(
        page_content="Tristana: Basic Attack Range, Damage, and Attack Speed Ratio increased. AD growth, HP Regen growth, and Armor growth decreased. Passive range decreased at higher levels. Q Attack Speed increased. W slow decreased and duration changed to a flat value. W AD ratio added. E damage profile reworked to value crit. R cooldown changed to a flat value. R stun added. R damage profile reworked to account for AD ratio.",
        metadata={"type": "champion"},
    ),
    Document(
        page_content="Vladimir: W healing effectiveness reduced against minions.",
        metadata={"type": "champion"},
    ),
    Document(
        page_content="Components: Our goal with the component changes is to more accurately hit their true gold-to-stat values. Early resources and regeneration is still intentionally overpriced in order to keep those components from overtaking the laning phase, but they shouldn't feel like such large downgrades when they're part of later game builds.",
        metadata={"type": "item"},
    ),
    Document(
        page_content="Dagger: Total Cost: 300 => 250\nFaerie Charm: Total Cost: 250 => 200\nSapphire Crystal:Total Cost: 350 => 300, Mana: 250 => 300",
        metadata={"type": "item"},
    ),
    Document(
        page_content="Fighter Items: We rebalanced fighter items with the three classes in mind: skirmishers, who need to be highly lethal and require finesse to find success, divers, who need to get through the front line to access their weaker targets, and juggernauts, who should be tanky enough to hit whoever's in front of them. With these goals in mind, we focused on balancing item power budget between durability—which there's a lot of on fighter items—and repeatable damage effects. We also wanted to ensure that some resist-heavy tank items remain solid picks for when fighter players need to take up tank duties to counter enemy team comps.",
        metadata={"type": "item"},
    ),
]


"""Load data to pinecone"""

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

PINECONE_INDEX_NAME = "docs-rag-chatbot"
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))

# Target index and check status
pc_index = pc.Index(PINECONE_INDEX_NAME)
print(pc_index.describe_index_stats())

embeddings = HuggingFaceEmbeddings()
namespace = "test"

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
