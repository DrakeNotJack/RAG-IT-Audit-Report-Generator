from pathlib import Path
import json
from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core._api.deprecation import LangChainDeprecationWarning

import warnings
import os

os.environ['LANGCHAIN_SUPPRESS_DEPRECATION_WARNING'] = '1'
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# ========= 1. Path configuration =========
# This file: rag_it_audit/src/build_index.py
BASE_DIR = Path(__file__).resolve().parent.parent   # points to rag_it_audit/
DATA_DIR = BASE_DIR / "data"
STANDARD_DIR = DATA_DIR / "standards"
POLICY_DIR = DATA_DIR / "policies"
CLIENT_DIR = DATA_DIR / "client_inquiries"

CHROMA_DIR = BASE_DIR / "chroma_db"  # vector DB persistence directory


# ========= 2. Load text files =========
def load_text_dir(dir_path: Path, base_metadata: dict) -> List[Document]:
    """
    Load all .txt files in a directory and convert them to a list of Documents.
    base_metadata is used to attach common metadata fields (for example, doc_type).
    """
    docs: List[Document] = []
    for path in dir_path.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        metadata = {
            "source": str(path.relative_to(BASE_DIR)),
            **base_metadata,
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def load_client_json(dir_path: Path) -> List[Document]:
    """
    Convert client JSON inquiries to a list of Documents.
    Each module becomes one Document whose content is the concatenation of its Q&A entries.
    """
    docs: List[Document] = []
    for path in dir_path.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        client_name = data.get("client_name", "Unknown Client")
        modules = data.get("modules", {})

        for module_key, module_data in modules.items():
            inquiries = module_data.get("inquiries", {})
            lines = []
            for topic, answer in inquiries.items():
                lines.append(f"{topic}: {answer}")
            content = "\n".join(lines)

            metadata = {
                "source": str(path.relative_to(BASE_DIR)),
                "client_name": client_name,
                "module": module_key,
                "doc_type": "client_inquiry",
            }
            docs.append(Document(page_content=content, metadata=metadata))

    return docs


# ========= 3. Main function: chunking + embedding + building the vector DB =========
def build_index():
    # 3.1 Load the three sets of documents
    standard_docs = load_text_dir(
        STANDARD_DIR,
        {"doc_type": "standard"},
    )
    policy_docs = load_text_dir(
        POLICY_DIR,
        {"doc_type": "policy"},
    )
    client_docs = load_client_json(CLIENT_DIR)

    all_docs = standard_docs + policy_docs + client_docs
    print(f"Loaded {len(all_docs)} documents in total")
    
    #print(all_docs)

    if not all_docs:
        print("X No documents were loaded. Please check the data/ directory structure and file contents.")
        return

    # 3.2 Chunking (split long documents into smaller pieces)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150, #Since testing documents are shorter than actual, use smaller chunk size and overlap
        length_function=len,
    )
    split_docs = splitter.split_documents(all_docs)
    print(f"There are {len(split_docs)} document chunks after splitting")

    # 3.3 Use a local embedding model (via Ollama)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 3.4 Build a Chroma vector store and persist it to disk
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vectordb.persist()
    print(f"✔ The vector DB has been saved to {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()