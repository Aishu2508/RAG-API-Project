from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

app = FastAPI()

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="mobile_collection"
)

# Mobile Dataset
mobiles = [
    {
        "name": "Samsung S25",
        "price": 85000,
        "camera": "200MP",
        "battery": "5000mAh",
        "ram": "12GB"
    },
    {
        "name": "OnePlus 13",
        "price": 70000,
        "camera": "50MP",
        "battery": "6000mAh",
        "ram": "16GB"
    },
    {
        "name": "iPhone 16",
        "price": 90000,
        "camera": "48MP",
        "battery": "4500mAh",
        "ram": "8GB"
    }
]

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

# Store Data in ChromaDB
for i, mobile in enumerate(mobiles):

    text = f"""
    Mobile Name: {mobile['name']}
    Price: {mobile['price']}
    Camera: {mobile['camera']}
    Battery: {mobile['battery']}
    RAM: {mobile['ram']}
    """

    chunks = splitter.split_text(text)

    for j, chunk in enumerate(chunks):

        embedding = embedding_model.encode(
            chunk
        ).tolist()

        try:
            collection.add(
                ids=[f"{i}_{j}"],
                embeddings=[embedding],
                documents=[chunk]
            )
        except:
            pass


@app.get("/")
def home():
    return {
        "message": "Mobile Recommendation System Running"
    }


@app.get("/view-data")
def view_data():

    data = collection.get()

    return {
        "ids": data["ids"],
        "documents": data["documents"]
    }


@app.get("/recommend")
def recommend_mobile(query: str):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    return {
        "User_Query": query,
        "Retrieved_Data": result["documents"]
    }