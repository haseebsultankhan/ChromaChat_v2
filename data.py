import os
import json
from glob import glob
from typing import Iterator, Tuple, Dict, Any, List

from tqdm import tqdm
import ollama
import chromadb


# --------------------------
# Configuration
# --------------------------
DATA_ROOT = "data-jsons"                 # root folder you described
DB_PATH = "chroma_bge_m3_db"             # ChromaDB directory
COLLECTION_NAME = "qa_collection"        # single collection
DIMENSION = 1024                         # bge-m3 embedding length


# --------------------------
# Helpers
# --------------------------
def embed_text(text: str) -> List[float]:
    """
    Get a 1024-dim embedding from Ollama bge-m3:latest.
    """
    resp = ollama.embeddings(model="bge-m3:latest", prompt=text)
    return resp["embedding"]


def iter_json_paths(root: str) -> Iterator[str]:
    """
    Yield all .json file paths recursively under root.
    """
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)


def extract_pairs(json_path: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """
    Yield (question, answer, meta) from a single JSON file.

    meta includes: domain, file_name (from JSON or filename fallback), source_path
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain = data.get("domain", "")
    file_name = data.get("file_name") or os.path.basename(json_path)
    qa_list = data.get("qa_data", [])

    for item in qa_list:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q or not a:
            continue
        yield q, a, {
            "domain": domain,
            "file_name": file_name,
            "source_path": json_path
        }


def count_total_pairs(root: str) -> int:
    total = 0
    for p in iter_json_paths(root):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            qa_list = d.get("qa_data", [])
            # Count only non-empty pairs
            total += sum(1 for it in qa_list if it.get("question") and it.get("answer"))
        except Exception:
            # Corrupt JSON or unexpected format – skip but keep going
            pass
    return total


# --------------------------
# Main build routine
# --------------------------
def main():
    # Init ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)

    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity like Milvus
    )

    # Get current count for incremental IDs  
    existing_count = collection.count()
    next_id = existing_count

    # Count total pairs for progress bar
    total_pairs = count_total_pairs(DATA_ROOT)
    print(f"Discovered ~{total_pairs} Q/A pairs under '{DATA_ROOT}'. Starting ingestion...")
    print(f"Collection currently has {existing_count} documents.")

    # We'll batch inserts for better speed
    batch_ids: List[str] = []
    batch_embeddings: List[List[float]] = []
    batch_documents: List[str] = []
    batch_metadatas: List[Dict[str, Any]] = []
    BATCH_SIZE = 64

    with tqdm(total=total_pairs, desc="Inserting Q/A pairs", unit="pair") as pbar:
        for json_path in iter_json_paths(DATA_ROOT):
            try:
                for question, answer, meta in extract_pairs(json_path):
                    # Embed the QUESTION (typical retrieval setup)
                    vec = embed_text(question)

                    # ChromaDB format
                    batch_ids.append(str(next_id))
                    batch_embeddings.append(vec)
                    batch_documents.append(question)  # Store question as document
                    batch_metadatas.append({
                        "answer": answer,
                        "domain": meta.get("domain", ""),
                        "file_name": meta.get("file_name", ""),
                        "source_path": meta.get("source_path", json_path),
                    })
                    
                    next_id += 1
                    pbar.update(1)

                    if len(batch_ids) >= BATCH_SIZE:
                        collection.add(
                            ids=batch_ids,
                            embeddings=batch_embeddings,
                            documents=batch_documents,
                            metadatas=batch_metadatas
                        )
                        batch_ids.clear()
                        batch_embeddings.clear()
                        batch_documents.clear()
                        batch_metadatas.clear()

            except Exception as e:
                # Keep ingesting even if one file has issues
                print(f"[WARN] Skipping '{json_path}': {e}")

        # Flush any remainder
        if batch_ids:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

    # Final stats
    final_count = collection.count()
    print(f"Done. Collection '{COLLECTION_NAME}' now has {final_count} documents.")


if __name__ == "__main__":
    main()