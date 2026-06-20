from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Configure

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect Weaviate
client = weaviate.connect_to_local()

# Create Collection
if not client.collections.exists("MobileCollection"):
    client.collections.create(
        name="MobileCollection",
        vectorizer_config=Configure.Vectorizer.none()
    )

collection = client.collections.get("MobileCollection")

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

for mobile in mobiles:

    text = f"""
    Mobile Name: {mobile['name']}
    Price: {mobile['price']}
    Camera: {mobile['camera']}
    Battery: {mobile['battery']}
    RAM: {mobile['ram']}
    """

    vector = embedding_model.encode(text).tolist()

    collection.data.insert(
        properties={
            "mobile_name": mobile["name"],
            "price": str(mobile["price"]),
            "camera": mobile["camera"],
            "battery": mobile["battery"],
            "ram": mobile["ram"],
            "document": text
        },
        vector=vector
    )

print("Data inserted into Weaviate successfully")

client.close()