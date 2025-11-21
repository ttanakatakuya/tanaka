# main.py — minimal RAG translator (Groq + sentence-transformers)
# Requirements: groq, sentence-transformers, pypdf, numpy
# Model: llama-3.1-8b-instant (Groq) — make sure GROQ_API_KEY is set

import os
import time
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

PDF = "sample.pdf"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
GROQ_MODEL = "llama-3.1-8b-instant"  # <-- use this exact name

# check API key
if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("Please set GROQ_API_KEY environment variable and restart the terminal.")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# load PDF pages (simple page-level extraction)
def load_pages(path):
    if not os.path.exists(path):
        raise SystemExit(f"Put your PDF as '{path}' in the project folder.")
    r = PdfReader(path)
    pages = []
    for i, p in enumerate(r.pages):
        t = p.extract_text() or ""
        t = t.strip()
        if t:
            pages.append(f"(page {i+1})\n" + t)
    return pages

# chunk (optional: here we keep page-level chunks)
def chunk_pages(pages):
    return pages  # page-level; change if you want finer chunks

# safe extract function for Groq response
def extract_content(resp):
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if content:
                return content
    except Exception:
        pass
    # fallback: try stringifying
    try:
        return str(resp)
    except Exception:
        return "[No content]"

# main
print("[INFO] Loading embedder:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

print("[INFO] Loading PDF...")
pages = load_pages(PDF)
if not pages:
    raise SystemExit("No text extracted from PDF.")
print(f"[INFO] {len(pages)} pages loaded.")

texts = chunk_pages(pages)
print(f"[INFO] {len(texts)} chunks prepared.")

print("[INFO] Building embeddings (fast)...")
doc_embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
print("[INFO] Embeddings ready.")

def query_to_answer(question, top_k=3):
    # embed query
    q_emb = embedder.encode([question], convert_to_numpy=True)
    # search
    hits = util.semantic_search(query_embeddings=q_emb, corpus_embeddings=doc_embs, top_k=top_k)[0]
    retrieved = [texts[h["corpus_id"]] for h in hits]
    context = "\n\n".join(retrieved)

    prompt = [
        {"role": "system", "content": "You are an academic translator. Translate the CONTEXT (French/German) into clear English, then answer the QUESTION using the translated context. Cite page numbers when relevant."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer in English."}
    ]

    # call Groq with the single chosen model
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=prompt,
            max_tokens=600,
            temperature=0.0
        )
        return extract_content(resp)
    except Exception as e:
        # log and fallback to returning context snippet
        print("[WARN] Groq call failed:", e)
        return "※Model unavailable — returning retrieved context snippet:\n\n" + context[:3000]

if __name__ == "__main__":
    print("RAG ready. Type questions (type 'exit' to quit).")
    while True:
        q = input("\nQ: ").strip()
        if not q or q.lower() in ("exit", "quit"):
            break
        ans = query_to_answer(q, top_k=3)
        print("\n--- Answer ---\n")
        print(ans)
        print("\n---------------\n")
