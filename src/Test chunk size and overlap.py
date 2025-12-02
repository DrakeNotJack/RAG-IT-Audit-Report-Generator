from pathlib import Path
from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ========= 1. Path configuration =========
# This file: rag_it_audit/src/build_index.py
BASE_DIR = Path(__file__).resolve().parent.parent   # points to rag_it_audit/
DATA_DIR = BASE_DIR / "data"
STANDARD_DIR = DATA_DIR / "standards"
POLICY_DIR = DATA_DIR / "policies"
CLIENT_DIR = DATA_DIR / "client_inquiries"

CHROMA_DIR = BASE_DIR / "chroma_db"  # vector DB persistence directory      

def validate_current_chunking(sample_text):
    """Verify whether the current chunk_size configuration is appropriate."""
    
    # Test the effect of different chunk_size values
    test_sizes = [400, 600, 800, 1000]
    
    for size in test_sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=int(size * 0.25),  # 25% overlap
            length_function=len,
        )
        
        chunks = splitter.split_text(sample_text)
        avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
        
        print(f"chunk_size={size}: generated {len(chunks)} chunks, average length {avg_length:.0f} characters")
        
        # Check semantic integrity
        complete_sentences = sum(1 for chunk in chunks 
                               if chunk.strip() and chunk.strip()[-1] in '.!?')
        print(f"  Full sentence ratio: {complete_sentences/len(chunks)*100:.1f}%")

for path in STANDARD_DIR.glob("*.txt"):
    text = path.read_text(encoding="utf-8")
    validate_current_chunking(text)    
    

#In Actual Practice will involve manual review