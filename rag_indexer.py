import os
import torch
from sentence_transformers import SentenceTransformer

# Setup folder paths
DOCS_DIR = "./rag_documents"
DB_PATH = "./vector_store.pt"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text, max_chars=300):
    """
    Splits text into readable chunks based on sentence boundaries.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_chars:
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentence]
            current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len + 2  # account for ". "
            
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
        
    return chunks

def build_index():
    if not os.path.exists(DOCS_DIR):
        print(f"Creating documents folder at: {DOCS_DIR}")
        os.makedirs(DOCS_DIR)
        print("Please place text (.txt) or markdown (.md) files in it and re-run this script!")
        return

    print("=== Step 1: Scanning rag_documents/ Folder ===")
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith((".txt", ".md"))]
    if not files:
        print("No .txt or .md files found in rag_documents/. Ingestion aborted.")
        return
        
    print(f"Found {len(files)} file(s): {files}")
    
    all_chunks = []
    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            print(f"  - Parsed {file_name}: split into {len(chunks)} chunks.")

    if not all_chunks:
        print("No text chunks could be extracted. Ingestion aborted.")
        return

    print(f"\n=== Step 2: Loading Local Embedding Model ({EMBEDDING_MODEL_ID}) ===")
    # Loads model onto CPU/GPU automatically
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
    
    print("\n=== Step 3: Generating Vector Embeddings ===")
    # Encode sentences into 384-dimensional coordinates
    embeddings = embedding_model.encode(all_chunks, convert_to_tensor=True)
    
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")
    
    print("\n=== Step 4: Saving Vector Store Database ===")
    torch.save({
        "chunks": all_chunks,
        "embeddings": embeddings.cpu() # Save vectors on CPU for lightweight retrieval
    }, DB_PATH)
    
    print(f"🎉 Success! Database saved locally to: {DB_PATH}")

if __name__ == "__main__":
    build_index()
