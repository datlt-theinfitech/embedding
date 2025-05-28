from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = None  

class SentencesRequest(BaseModel):
    sentences: List[str]

@app.on_event("startup")
async def load_model():
    global model
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model = SentenceTransformer(model_name)
    print("Model loaded successfully at startup.")

@app.post("/embed")
def embed_sentences(req: SentencesRequest):
    if model is None:
        return {"error": "Model not loaded yet."}
    embeddings = model.encode(req.sentences)
    return {"embeddings": embeddings.tolist()}

