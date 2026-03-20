import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_SIZE = 500


def load_pdfs(folder="data"):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts


def chunk_text(texts):
    chunks = []
    for text in texts:
        for i in range(0, len(text), CHUNK_SIZE):
            chunks.append(text[i:i+CHUNK_SIZE])
    return chunks


def create_index():
    texts = load_pdfs()
    chunks = chunk_text(texts)

    embeddings = MODEL.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, chunks


def search(query, index, chunks, k=3):
    query_vec = MODEL.encode([query])
    distances, indices = index.search(np.array(query_vec), k)

    results = [chunks[i] for i in indices[0]]
    return results