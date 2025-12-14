# 🏢 RAG-powered IT Audit Report Generator (Prototype)
Interact via natural language queries, and generate professional audit reports with one click.

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-%F0%9F%A6%9C%F0%9F%94%97-orange.svg)

## 📌 Demo Project Note
This prototype explores how IT audit workflows can be streamlined using Retrieval-Augmented Generation (RAG). It adopts the Ollama local ecosystem (nomic-embed-text for embeddings and Qwen 2.5-7B for LLM inference) for the following reasons:
- Local & free deployment: No online API calls, zero cost
- Low hardware threshold: Runs on consumer-grade GPUs (≥8GB VRAM) or 16GB RAM (CPU inference)
- Fast iteration: Enables quick verification of the RAG audit pipeline without complex configuration

For production environments, refer to the "Key Technical Decisions" section for scalable, privacy-compliant upgrades (e.g., locally/privately deployed large models).

## 📋 Table of Contents
- [🧩 Business Motivation](#-business-motivation)
- [🔧 Technical Architecture](#-technical-architecture)
- [📁 Project Structure](#-project-structure)
- [🤔 FAQ](#-faq)
  
## 🧩 Business Motivation
Internal audit work involves a large amount of repetitive, document-intensive tasks, such as searching for standard clauses, comparing them with clients' practical processes, and organizing evidence. These steps are time-consuming but low-value, often occupying most auditors' time.

This project is to release auditors from those cumbersome routine tasks, allowing them to focus on works requiring real judgment and analysis (such as risk assessment, control design evaluation, management discussions, etc.).

While the examples come from ITGC auditing, the core concept applies to any "process analysis + document alignment" scenario.

## 🔧 Technical Architecture
### Tech Stack
- LangChain + LCEL - For maintainable AI pipeline construction
- ChromaDB - Vector database with rich metadata filtering capabilities
- Ollama - Local LLM runtime environment
- nomic-embed-text - Embedding model balancing retrieval quality and deployment cost
- qwen2.5:7b - Generation model balancing speed and quality
  
### Key Technical Decisions
#### 1. Embedding Model - Why nomic-embed-text
- Better for long texts: Standards, policies, and process documents are typically lengthy
- Local execution: Protects client data privacy (via Ollama)
- Higher retrieval accuracy: Better performance than default Ollama embeddings for domain-specific content

> **Note:** In production environments with stronger GPU support, this can be extended to larger embedding models (e.g., BGE-large) or hybrid setups (base embedding + reranker) if evaluation shows additional retrieval gains are needed.

#### 2. Vector Database - Why ChromaDB
Audit documents rely heavily on metadata (doc_type/module/client_name). Chroma supports:
- Persistent storage - Avoids rebuilding indexes each time
- Powerful metadata filtering - Precise retrieval by client, module, document type
- Scalability - Suitable for expanding evidence libraries as new documents are added

#### 3. Why LCEL
The audit process itself is chain-structured (question → extract key points → retrieve evidence → generate report). LCEL makes the entire RAG pipeline:
- Easier to debug - Each step can be tested independently
- More transparent - Meets audit traceability requirements
- More extensible - Easy to add quality checks, reranking, modularization, etc.

#### 4. Why Qwen 2.5-7B
For this demo, its strengths lie in practicality and compact-model performance:
- Strong audit scenario fit: Outperforms peers (Llama 3.1 8B, DeepSeek R1 7B) in bilingual (CN/EN) understanding and structured audit output
- Ollama-native integration: Enables rapid RAG workflow validation with minimal configuration
- High dev efficiency: Lowers testing thresholds, aligning with demo prototype goals

> **Note:** For production, upgrade to locally/privately deployed large models (e.g., DeepSeek-V2 local, Llama 3 private) for enhanced complex scenario performance (nuanced compliance/risk analysis) while ensuring audit data security.

#### 5. Multi-layer Quality Control
##### Chunking Configuration (Scenario-Adapted & Tunable)
Audit documents are lengthy with tight structure. Recommended parameter range (adjustable based on document type):
- chunk_size = 1000–1200 (balances context integrity and retrieval precision)
- chunk_overlap = 150–250 (preserves logical continuity for cross-chunk content)
- top_k = 8 (optimal number of retrieved chunks for audit report generation)

##### Retrieval Layer Precision
- Metadata filtering (by doc_type/module/client) + domain keyword boosting

##### Generation & Hallucination Mitigation
- Prompt constraints: Strict structured output requirements + mandatory audit evidence citation
- Anti-hallucination rule: Explicit prompt instruction – "Only generate content based on retrieved evidence"

##### Production Environment Extension
- Post-Retrieval Refinement: Reranking module for complex audit scenarios
- Quality Assurance: Human-in-the-loop validation sampling
- Performance Tuning: Feedback-driven parameter optimization

## 📁 Project Structure
```plaintext
RAG for IT Audit/
├── data/                    # Data directory (excluded from version control)
│   ├── standards/          # ITGC standard documents
│   ├── policies/           # Company policy documents
│   └── client_inquiries/   # Client inquiry responses
├── src/                    # Source code
│   ├── build_index.py      # Build vector index
│   └── generate_report.py  # Generate audit reports
├── chroma_db/              # Vector database storage (auto-generated)
├── reports/                # Generated reports (auto-generated)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore file
```

## 🤔 FAQ
Q: How much RAM is required?                           
A: Minimum 16GB RAM recommended. 7B model requires ~8GB, plus system and vector database overhead.                 
Q: Can it be extended to other audit domains?                                    
A: Yes. Simply:                        
Add the new domain’s standards, policies and client data to the data/ folder (follow the existing structure: standards/, policies/, client_inquiries/);
Update (or fine-tune) the parsing, retrieval, and report prompt templates as needed.

---
*Let AI automate the mundane, humans drive the strategy.*
