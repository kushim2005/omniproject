# OmniBrain - Agentic Multi-Modal RAG Orchestrator

OmniBrain is an **Agentic Multi-Modal Retrieval-Augmented Generation (RAG) system** designed to answer user queries from uploaded documents using intelligent retrieval, specialized agents, self-correction, and safety guardrails.

The system combines **FastAPI, LangGraph, FAISS/Qdrant, Sentence Transformers, Vision Language Models, Self-RAG, Query Rewriting, and NeMo Guardrails** into an end-to-end intelligent document question-answering pipeline.

---

# 📌 Project Objective

Traditional RAG systems can fail when:

- Documents contain both text and images
- User queries are ambiguous
- Retrieved information is irrelevant
- Generated answers contain unsupported information
- Users ask questions outside the system's intended scope

OmniBrain addresses these limitations by introducing:

- Multi-modal document ingestion
- Semantic vector search
- Agent-based routing
- Specialized subagents
- Query rewriting
- Retrieval evaluation
- Answer evaluation
- Self-RAG correction loops
- NeMo Guardrails for input validation

---

# 🗓️ Project Timeline

## Week 1 - Multi-Modal Document Ingestion & Vector Database

### Objective

Build the foundation for processing documents and converting their contents into searchable representations.

### Work Completed

- PDF document parsing
- Text extraction using PyMuPDF
- Page-level text processing
- Text cleaning and preprocessing
- Text chunking
- PDF image extraction
- Image processing
- Text embeddings generation
- Image embeddings generation
- Vector database preparation
- Semantic similarity search
- FastAPI backend setup
- Document upload API
- Document deletion API
- Chat API
- Health-check API

### Technologies

- Python
- FastAPI
- PyMuPDF
- Sentence Transformers
- FAISS
- Qdrant
- OpenCLIP
- NumPy

### Week 1 Pipeline

```text
PDF Document
     |
     v
PDF Parsing
     |
     +------------------+
     |                  |
     v                  v
   Text               Images
     |                  |
     v                  v
Text Cleaning      Image Processing
     |                  |
     v                  v
Text Chunking      Image Embeddings
     |                  |
     v                  |
Text Embeddings <----+
     |
     v
Vector Database
# Week 2 - Agentic RAG & LangGraph Integration

## Objective

The objective of Week 2 was to transform the basic RAG pipeline developed in Week 1 into an **agentic multi-modal RAG system** using LangGraph. The system was designed to intelligently route user queries to specialized agents based on the type of information requested.

## Week 2 Work Completed

- Implemented LangGraph state machine
- Designed workflow nodes and transitions
- Implemented Supervisor Agent
- Implemented query routing logic
- Developed specialized PDF/Search Agent
- Developed Vision Agent
- Implemented General/Greeting Agent
- Integrated agents with the vector database
- Connected LangGraph workflow with FastAPI backend
- Developed Streamlit chat interface
- Added PDF upload functionality
- Integrated chat-based querying
- Added conversation handling
- Implemented workflow/thought-process visualization
- Performed end-to-end integration testing

## LangGraph Workflow

The LangGraph workflow acts as the orchestration layer of OmniBrain. It receives the user's query and determines which specialized agent should process the request.

```text
                    User Query
                        |
                        v
                 FastAPI Backend
                        |
                        v
              LangGraph Supervisor
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     PDF/Search      Vision        General
       Agent          Agent         Agent
          |             |             |
          v             v             |
      Vector DB      Image Data       |
          |             |             |
          +-------------+-------------+
                        |
                        v
                   Final Answer
                        |
                        v
                  Streamlit UI

# Week 3 - Self-RAG, Query Rewriting & NeMo Guardrails

## Objective

The objective of Week 3 was to improve the reliability, accuracy, self-correction, and safety of the OmniBrain Agentic RAG system.

The system was enhanced with:

- Self-RAG
- Query Rewriting
- Retrieval Evaluation
- Answer Evaluation
- Confidence Scoring
- Iterative Self-Correction
- NeMo Guardrails
- Component-level testing and validation

These components allow OmniBrain to evaluate its own retrieval and generated answers before returning a final response to the user.

---

## Week 3 Work Completed

### 1. Self-RAG Implementation

Implemented a Self-RAG mechanism that evaluates the quality of the generated answer and determines whether another retrieval/generation iteration is required.

The Self-RAG process includes:

- Query processing
- Retrieval
- Retrieval evaluation
- Answer generation
- Answer evaluation
- Confidence scoring
- Retry mechanism
- Maximum iteration control

### Self-RAG Workflow

```text
                 User Query
                     |
                     v
              Query Rewriting
                     |
                     v
               Retrieval
                     |
                     v
          Retrieval Evaluation
                     |
                     v
              Answer Generation
                     |
                     v
            Answer Evaluation
                     |
              +------+------+
              |             |
           Accepted       Rejected
              |             |
              v             v
        Final Answer    Self-Correction
                            |
                            v
                      Query Rewrite
                            |
                            └──────> Retrieval

---

# Week 4 - Evaluation, Observability & UI Refinement

## Objective

The objective of Week 4 was to evaluate the reliability and quality of the OmniBrain RAG pipeline and prepare the system for final refinement. The focus was on evaluating retrieval and answer quality, improving system observability, and enhancing the user interface with transparent citation support.

---

## Week 4 Work Completed

### 1. RAG Evaluation

The Self-RAG pipeline was evaluated using retrieval and answer-quality metrics.

The evaluation focused on:

- Retrieval relevance
- Retrieval confidence
- Answer confidence
- Answer groundedness
- Answer completeness
- Hallucination score
- Number of Self-RAG iterations

These metrics help determine whether the retrieved documents contain sufficient information and whether the generated response is properly grounded in the retrieved context.

### 2. Self-RAG Validation

The Self-RAG workflow was validated using document-based queries.

The validation confirmed that:

- Relevant documents can be retrieved from the FAISS vector database.
- Retrieved documents are evaluated before answer generation.
- Answers can be generated using retrieved document content.
- Answer groundedness and confidence can be evaluated.
- The system can determine whether an answer should be accepted or regenerated.
- The system provides a fallback response when sufficient supporting context is unavailable.

### 3. Observability & Evaluation Metrics

The project was prepared for LLM observability and performance evaluation using Langfuse.

The planned monitoring metrics include:

- Token usage
- LLM execution traces
- Response latency
- Model execution details
- RAG pipeline performance

These metrics can be used to analyse the efficiency and behaviour of the LLM-based components.

### 4. Citation & Explainability

The final UI refinement focused on improving transparency of AI-generated responses through citation support.

The planned citation functionality allows users to:

- Identify the source document used for an answer.
- Navigate to the relevant PDF page.
- Trace an AI-generated claim back to its source.
- Reference charts or visual information associated with the retrieved content.

This improves the explainability and trustworthiness of the RAG system.

### 5. Final UI Refinement

The user interface was prepared for final refinement to provide:

- Clear AI responses
- Source references
- PDF-based evidence
- Citation links
- Improved response transparency
- Better user interaction with retrieved information

---

## Week 4 Pipeline

```text
                    User Query
                        |
                        v
                 Self-RAG Pipeline
                        |
                        v
                  Document Retrieval
                        |
                        v
                Retrieval Evaluation
                        |
                        v
                 Answer Generation
                        |
                        v
                  Answer Evaluation
                        |
             +----------+----------+
             |                     |
             v                     v
       Quality Metrics        Final Answer
             |                     |
             v                     v
     Groundedness              Citations
     Completeness                 |
     Confidence                   v
     Hallucination            PDF Source
             |                     |
             +----------+----------+
                        |
                        v
                Final User Interface
                        |
                        v
              Observability / Metrics
                   (Langfuse)