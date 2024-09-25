from pathlib import Path

import faiss
import numpy as np
import pymupdf
from wordllama import WordLlama

embedding_model = WordLlama.load(dim=1024)
embedding_db = np.empty((0, 1024), dtype=np.float32)

file_path = Path("data/Don Quixote-www.learnenglishteam.com.pdf")
with pymupdf.open(file_path) as pdf:
    for page in pdf:
        content = page.get_text()
        embedding = embedding_model.embed(content)
        embedding_db = np.append(embedding_db, embedding, axis=0)

embedding_index = faiss.IndexFlatL2(1024)
embedding_index.add(embedding_db)
faiss.write_index(embedding_index, "embedding.index")
