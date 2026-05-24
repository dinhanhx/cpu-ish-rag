from pathlib import Path

import faiss
import numpy as np
import pymupdf
from wordllama import WordLlama

SCRIPT_DIR = Path(__file__).resolve().parent

embedding_model = WordLlama.load(dim=1024)
embeddings = []

file_path = SCRIPT_DIR / "data/Don Quixote-www.learnenglishteam.com.pdf"
with pymupdf.open(file_path) as pdf:
    for page in pdf:
        content = page.get_text()
        embedding = embedding_model.embed(content)
        embeddings.append(embedding)

embedding_db = np.vstack(embeddings).astype(np.float32)

embedding_index = faiss.IndexFlatL2(1024)
embedding_index.add(embedding_db)
faiss.write_index(embedding_index, str(SCRIPT_DIR / "embedding.index"))
