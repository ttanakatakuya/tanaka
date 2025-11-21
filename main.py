# main.py — robust version (fixes ChatCompletionMessage subscript error)
import os
import time
import traceback
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

# ---- config ----
PDF_PATH = "sample.pdf"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
PREFERRED_MODELS = ["llama-3.1-8b-instant", "qwen/qwen3-32b", "groq/compound-mini"]  # note: first has no "groq/" prefix

# Check API key
if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("Set GROQ_API_KEY environment variable (setx GROQ_API_KEY \"your_key\" then restart shell).")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- helpers ----
def safe_extract_content(resp):
    """
    Safely extract text content from Groq chat completion response object.
    Tries several access patterns and falls back to stringification.
    """
    # Normalize common container locations
    choices = None
    try:
        choices = getattr(resp, "choices", None)
    except Exception:
        choices = None

    if choices is None:
        # some SDKs put results under `data`
        try:
            choices = getattr(resp, "data", None)
        except Exception:
            choices = None

    if choices:
        # get the first item in a robust way
        first = None
        try:
            if isinstance(choices, (list, tuple)):
                first = choices[0]
            elif isinstance(choices, dict):
                # take first value
                first = next(iter(choices.values()))
            else:
                # fallback: try indexing
                first = choices[0]
        except Exception:
            first = choices

        # Try common shapes on the first choice
        try:
            # 1) .message.content or .message (string/dict)
            msg = getattr(first, "message", None)
            if msg is not None:
                if isinstance(msg, str):
                    return msg
                content = getattr(msg, "content", None)
                if content:
                    if isinstance(content, (list, tuple)):
                        return " ".join(str(c) for c in content)
                    return str(content)
                if isinstance(msg, dict):
                    if "content" in msg:
                        return str(msg["content"])
                    if "text" in msg:
                        return str(msg["text"])

            # 2) direct .content attribute
            content = getattr(first, "content", None)
            if content:
                return str(content)

            # 3) dict-like first choice
            if isinstance(first, dict):
                # try nested message->content
                msg = first.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg["content"])
                # try a few common top-level keys
                for k in ("content", "text", "output"):
                    v = first.get(k)
                    if isinstance(v, str):
                        return v
                    if isinstance(v, (list, tuple)):
                        return " ".join(str(x) for x in v)

            # 4) fallback to stringifying the first choice
            return str(first)
        except Exception:
            try:
                return str(first)
            except Exception:
                pass

    # Try top-level text-like fields
    for key in ("text", "output_text", "content"):
        try:
            v = getattr(resp, key, None)
            if v:
                return str(v)
        except Exception:
            continue

    # Last resort: stringify the whole response
    try:
        return str(resp)
    except Exception:
        raise RuntimeError("Unable to extract content from model response")


def get_available_models():
    """
    Query Groq for active, usable model ids. Return ordered list preferring PREFERRED_MODELS.
    """
    try:
        models_resp = client.models.list()
        models = getattr(models_resp, "data", models_resp)
    except Exception:
        # if listing fails, fallback to preferred
        return PREFERRED_MODELS.copy()

    ids = []
    for m in models:
        try:
            mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
            active = (m.get("active") if isinstance(m, dict) else getattr(m, "active", False))
            max_tok = (m.get("max_completion_tokens") if isinstance(m, dict) else getattr(m, "max_completion_tokens", 0))
            if not mid or not active or not (isinstance(max_tok, (int, float)) and max_tok > 0):
                continue
            low = mid.lower()
            if any(bad in low for bad in ("guard", "tts", "prompt-guard", "safeguard")):
                continue
            ids.append(mid)
        except Exception:
            continue

    ordered = []
    for p in PREFERRED_MODELS:
        # prefer exact match
        if p in ids:
            ordered.append(p)
            continue
        # if preferred name lacks groq/ prefix but a prefixed id exists, prefer that
        if not p.startswith("groq/") and ("groq/" + p) in ids:
            ordered.append("groq/" + p)
            continue
        # if preferred has groq/ but server lists without prefix, accept that too
        if p.startswith("groq/") and p[5:] in ids:
            ordered.append(p[5:])
            continue
    for idm in ids:
        if idm not in ordered:
            ordered.append(idm)
    if not ordered:
        # fall back to whatever ids we found or the hardcoded preferred list
        return ids or PREFERRED_MODELS.copy()
    return ordered


def try_call_model(model_name, messages, max_tokens=512, temperature=0.2):
    # Try once with the given model name, and if it fails and the name
    # doesn't start with 'groq/', retry with that prefix (common mismatch).
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return safe_extract_content(resp)
    except Exception:
        # retry with groq/ prefix if appropriate
        if not str(model_name).startswith("groq/"):
            try:
                alt = "groq/" + model_name
                resp = client.chat.completions.create(
                    model=alt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return safe_extract_content(resp)
            except Exception:
                pass
        raise

# ---- PDF load & chunking ----
def load_pages(path):
    if not os.path.exists(path):
        raise SystemExit(f"PDF not found: {path}")
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        txt = txt.strip()
        if txt:
            pages.append(f"(page {i+1})\n{txt}")
    return pages


def chunk_texts(pages, chunk_size=1000, overlap=200):
    chunks = []
    for pg in pages:
        L = len(pg)
        if L <= chunk_size:
            chunks.append(pg)
            continue
        start = 0
        while start < L:
            end = min(start + chunk_size, L)
            chunks.append(pg[start:end])
            start = end - overlap
            if start < 0:
                start = 0
            if start >= L:
                break
    return chunks

# ---- embedding ----
print("[INFO] Loading embedder:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading PDF...")
pages = load_pages(PDF_PATH)
if not pages:
    raise SystemExit("No text extracted from PDF.")
print(f"[INFO] {len(pages)} pages loaded.")

texts = chunk_texts(pages, chunk_size=1000, overlap=200)
print(f"[INFO] {len(texts)} chunks prepared.")

print("[INFO] Computing embeddings (will be fast with MiniLM-L3)...")
doc_embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
print("[INFO] Embeddings ready.")

# ---- RAG ----
def rag_query(question, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    # semantic_search expects 2D arrays
    hits = util.semantic_search(query_embeddings=q_emb, corpus_embeddings=doc_embs, top_k=top_k)[0]
    retrieved = [texts[h["corpus_id"]] for h in hits]
    context = "\n\n".join(retrieved)

    prompt = [
        {"role": "system", "content": "You are a careful academic translator. Translate the CONTEXT (French or German) into clear academic English, then answer the QUESTION using the translated context. Cite page numbers when relevant."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer in English."}
    ]

    # try available models in order
    candidates = get_available_models()
    print("[INFO] Model candidates:", candidates)
    last_exc = None
    for model in candidates:
        print("[INFO] Trying model:", model)
        try:
            out = try_call_model(model, prompt, max_tokens=512, temperature=0.2)
            return out
        except Exception as e:
            print("[WARN] model failed:", model, "->", e)
            last_exc = e
            time.sleep(0.2)
            continue

    # fallback: return context snippet
    print("[ERROR] All models failed; returning context snippet as fallback.")
    return "※Model unavailable — returning retrieved context snippet:\n\n" + context[:3000]


if __name__ == "__main__":
    print("Ready — ask a question (empty to exit).\n")
    try:
        while True:
            q = input("\nQuestion> ").strip()
            if not q:
                print("Exiting.")
                break
            try:
                ans = rag_query(q, top_k=3)
                print("\nAnswer:\n", ans)
            except KeyboardInterrupt:
                raise
            except Exception:
                print("[ERROR] Exception during query:\n", traceback.format_exc())
    except KeyboardInterrupt:
        print("\nInterrupted.")
