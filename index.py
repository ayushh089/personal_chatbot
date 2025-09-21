from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

# Paths
data_dir = Path(__file__).parent / "app" / "data"
files = ["bio.txt", "resume.txt"]

# Load all documents
docs = []
for file in files:
    file_path = data_dir / file
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs.extend(loader.load())

print(f"Loaded {len(docs)} documents")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
chunks = text_splitter.split_documents(documents=docs)

# Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
)

# Qdrant Cloud configuration
QDRANT_CLOUD_URL = os.getenv("QDRANT_LINK")  # e.g., "https://xyz-123.qdrant.cloud"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")      # Your Qdrant Cloud API key

# Upload to Qdrant Cloud
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY,
    collection_name="about_ayush",
    distance="Cosine"  # optional, default is Cosine
)

print("Documents uploaded to Qdrant Cloud successfully!")
