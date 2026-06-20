from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import weaviate

# APP INIT (FIXED)
app = FastAPI()

# EMBEDDING MODEL
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# WEAVIATE CONNECTION
client = weaviate.connect_to_local()
collection = client.collections.get("MobileCollection")


# HOME
@app.get("/")
def home():
    return {
        "message": "RAG Mobile Recommendation System Using Weaviate"
    }


# RAG ENDPOINT
@app.get("/recommend")
def recommend_mobile(query: str):

    # Convert query to vector
    query_vector = embedding_model.encode(query).tolist()

    # Retrieve top results
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=3
    )

    # Containers
    all_chunks_created = {}
    retrieved_chunks = {}
    ranked_mobiles = []

    models = ["tinyllama", "qwen2.5:0.5b", "smollm2:360m"]
    scores = [95, 90, 85]

    # PROCESS RESULTS
    
    for idx, obj in enumerate(response.objects):

        print("OBJECT:", obj.properties)  # DEBUG (optional)

        mobile = obj.properties
        chunk_text = mobile.get("document", "")

        # ALL CHUNKS
        all_chunks_created[f"chunk{idx+1}"] = chunk_text

        # RETRIEVED CHUNKS
        retrieved_chunks[f"retrieved_chunk{idx+1}"] = chunk_text

        # RANKING LOGIC
        ranked_mobiles.append({
            "model_name": models[idx],
            "mobile_name": mobile["mobile_name"],
            "price": mobile["price"],
            "camera": mobile["camera"],
            "battery": mobile["battery"],
            "ram": mobile["ram"],
            "score": scores[idx],
            "model_suggestion": (
                f"{mobile['mobile_name']} is recommended because it provides "
                f"{mobile['camera']} camera, {mobile['ram']} RAM and "
                f"{mobile['battery']} battery."
            )
        })

    # Sort by score
    ranked_mobiles.sort(key=lambda x: x["score"], reverse=True)

    best_model = ranked_mobiles[0]

    # FINAL RESPONSE
    return {
        "user_query": query,
        "all_chunks_created": all_chunks_created,
        "retrieved_chunks": retrieved_chunks,
        "top_most_recommendation": best_model,
        "model_recommendation": best_model["model_suggestion"],
        "total_suggestions": len(ranked_mobiles),
        "all_ranked_suggestions": ranked_mobiles
    }