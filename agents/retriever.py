# agents/retriever.py
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Optional, Type
from pydantic import BaseModel
from crewai import Agent
from crewai.tools import BaseTool
from tools import RetrieveDataToolSchema
from utils.security import sanitize_input, encrypt, decrypt
from utils.db import get_all_decrypted_docs

# optional vector store
_use_vectorstore = False
_vectorstore = None
_embeddings = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _use_vectorstore = True
except Exception as e:
    print(f"[retriever] FAISS/embeddings unavailable; will fallback to TF-IDF. {e}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def _build_index():
    global _vectorstore, _embeddings
    docs = get_all_decrypted_docs()
    if not docs:
        return None
    texts = [f"{d['crop']} {d['location']} {d['data']}" for d in docs]
    if _use_vectorstore and _embeddings:
        try:
            from langchain.schema import Document
            ldocs = [Document(page_content=t) for t in texts]
            _vectorstore = FAISS.from_documents(ldocs, _embeddings)
            print("[retriever] Built FAISS vectorstore.")
            return _vectorstore
        except Exception as e:
            print("[retriever] building FAISS failed, falling back:", e)
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    _vectorstore = {"vectorizer": vec, "matrix": X, "texts": texts}
    print("[retriever] Built TF-IDF index fallback.")
    return _vectorstore

class RetrieveDataTool(BaseTool):
    name: str = "RetrieveData"
    description: str = "Retrieve relevant crop docs from DB"
    args_schema: Optional[Type[BaseModel]] = RetrieveDataToolSchema

    def _run(self, query: Optional[dict] = None) -> str:
        global _vectorstore
        if _vectorstore is None:
            _build_index()
        q = query.get("description") if isinstance(query, dict) else query
        if isinstance(q, str) and q.startswith("gAAAAA"):
            try:
                q = decrypt(q)
            except:
                pass
        q = sanitize_input(str(q or ""))
        if not q:
            return encrypt("No query provided.")
        # FAISS path (if available)
        try:
            if hasattr(_vectorstore, "similarity_search"):
                results = _vectorstore.similarity_search(q, k=3)
                raw = [r.page_content for r in results]
                return encrypt(f"Retrieved (FAISS): {raw}")
        except Exception:
            pass
        # TF-IDF fallback
        try:
            vec = _vectorstore["vectorizer"]
            X = _vectorstore["matrix"]
            qv = vec.transform([q])
            sims = cosine_similarity(qv, X).flatten()
            idx = np.argsort(-sims)[:3]
            docs = get_all_decrypted_docs()
            out = []
            for i in idx:
                if i < len(docs):
                    d = docs[i]
                    out.append(f"{d['crop']}|{d['location']}|{d['data'][:200]}")
            return encrypt(f"Retrieved (TFIDF): {out}")
        except Exception as e:
            print("[retriever] retrieval error:", e)
            return encrypt("Retrieval error.")

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Fetch and refine crop data securely and ethically",
    backstory="Integrates IR, NLP with Responsible AI checks",
    tools=[RetrieveDataTool()]
)

if __name__ == "__main__":
    from utils.security import decrypt
    test = {"description": "wheat yield in East"}
    enc = retriever_agent.tools[0]._run(test)
    print("Retrieved:", decrypt(enc))
