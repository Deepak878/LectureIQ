from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

# Environment Variables
load_dotenv()
url = "https://78f17131-c8cb-4499-a8ec-2449ed70b7d8.sa-east-1-0.aws.cloud.qdrant.io"
apikey = os.getenv("QDRANT_API_KEY")
collection_name = "appDB1"

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    url = url, 
    api_key = apikey,
)

# print(qdrant_client.get_collections())
# print(apikey)

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def vectorize(text, lecture_id):
    # Split text using lang chain text splitter
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)

        # 3. GENERATE ALL EMBEDDINGS AT ONCE (Batching)
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = model.encode(chunks)

        # Prepare points with payload
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": chunk,
                    "lecture_id": lecture_id
                }
            ) for chunk, emb in zip(chunks, embeddings)
        ]
    
        operation_info = qdrant_client.upsert(
            collection_name=collection_name, 
            points=points,
            wait=True  # Crucial for ensuring data is searchable immediately
        )
        
        return {
            "status": "success", 
            "chunks": len(chunks), 
            "lecture_id": lecture_id,
            "op_info": str(operation_info.status)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


text = """
The Solar System is a vast neighborhood in space, bound together by the immense gravitational pull of the Sun. At its heart lies a yellow dwarf 
star that accounts for more than 99% of the total mass of the entire system. This star provides the light and heat necessary to sustain life on Earth, 
while its gravity keeps everything from the smallest dust motes to the largest planets in a predictable, looping dance.
Would you like me to show you how to set up the Qdrant Search function now so you can ask questions like "What are the inner planets?" and find the answer in this text?
Would you like me to show you how to set up the Qdrant Search function now so you can ask questions like "What are the inner planets?" and find the answer in this text?
Would you like me to show you how to set up the Qdrant Search function now so you can ask questions like "What are the inner planets?" and find the answer in this text?
"""

result = vectorize(text, 3)
print(result)
# # 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)

# # 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)