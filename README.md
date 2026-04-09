# Prompt Similarity \& Deduplication Service

A production-style FastAPI service for **semantic search, prompt deduplication, and clustering** using embedding models and vector search (FAISS).
 

# &#x20;Features

* Generate embeddings for prompt templates
* Semantic search over prompt library
* Find similar prompts by ID
* Detect duplicate / near-duplicate prompts
* FAISS-based vector retrieval (fast cosine similarity)
* Dockerized deployment
* CI pipeline (GitHub Actions)
* Test scaffold (pytest)
 

# Architecture Overview

```
Prompt → Embedding → FAISS ANN → Top-K neighbors → Graph edges → Connected Components → Clusters
``` 

### Core Components

|Layer|Responsibility|
|-|-|
|API Layer|REST endpoints (FastAPI)|
|Service Layer|Similarity + embedding logic|
|Vector Store|FAISS index + metadata mapping|
|Embedding Model|SentenceTransformer (MiniLM)|
 

# Key Design Decisions

## 1\. Embedding Model Choice

We use:

`all-MiniLM-L6-v2`

Why:

* Lightweight (\~80MB)
* Fast inference
* Strong semantic similarity performance
* Suitable for 1k–1M prompt scale (with FAISS)

 

## 2\. Prompt Normalization 

Template variables like:

```
{{question\_text}}
{{user\_name}}
```

are normalized into:

```
<VAR>
```

This ensures:

* Similar prompts with different variable names cluster together
* Prevents embedding noise from template placeholders

 
 Another choice is to use Dual Embedding Strategy (furture work) which embeds BOTH:

* Raw prompt
* Normalized prompt

Then combine:

final_embedding = 0.7 * normalized + 0.3 * raw

This preserves:

*Structure similarity
* Content nuance

## 3\. Similarity Metric

We use **cosine similarity**:

* Implemented via dot product on normalized embeddings
* Efficient for FAISS IndexFlatIP

 
## 4\. Vector Store Choice (FAISS)

Why FAISS:

* Fast nearest neighbor search (O(log n) / approximate options)
* Works in-memory (low latency)
* Easy upgrade path to IVF/HNSW indexes
 
## 5\. Duplicate Detection Strategy

We use a **graph-based clustering approach**:

* Nodes = prompts
* Edge exists if similarity > threshold
* Connected components = duplicate clusters

This avoids:

* Pairwise manual inspection
* O(n²) runtime at scale (can be optimized later with FAISS filtering)


## 6\. Service Separation

We separate:

* EmbeddingService → model logic
* VectorStore → retrieval logic
* SimilarityService → business logic
* ClusteringService → analysis logic

This makes the system:

* Testable
* Swappable (FAISS → Pinecone / Weaviate)
* Maintainable
 

# Setup Instructions

## 1\. Clone \& Install

```bash
git clone https://github.com/meizhuliu/prompt-similarity-service.git
cd prompt-similarity-service
pip install -r requirements.txt
```
 

## 2\. Run API locally

```bash
uvicorn app.main:app --reload
```

Open:

```
http://localhost:8000/docs
```
 

## 3\. Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```
 
## 4\. Run Tests

```bash
pytest -q
```
 

# API Endpoints

## 1\. Generate Embeddings

```
POST /api/embeddings/generate
```

### Body

```json
{
  "prompts": \[
    {
      "prompt\_id": "example.id",
      "content": "Ask the user their name"
    }
  ]
}
```
 

## 2\. Semantic Search

```
POST /api/search/semantic
```

### Body

```json
{
  "query": "how to handle user confusion",
  "limit": 10
}
```
 

## 3\. Similar Prompts

```
GET /api/prompts/{prompt\_id}/similar
```

Query params:

* `limit`
* `threshold`
 

# Testing Strategy

Current tests include:

* API health validation
* Endpoint response checks

Planned improvements:

* Embedding consistency tests
* Similarity ranking validation
* Clustering correctness tests
 

# Performance Characteristics

|Scale|Behavior|
|-|-|
|1k prompts|Instant (in-memory)|
|10k prompts|Fast (FAISS IndexFlatIP)|
|100k+ prompts|Upgrade to IVF/HNSW recommended|

 

# Future Improvements

### 1\. Scalable Vector DB

* Replace FAISS with:

  * Pinecone
  * Weaviate
  * pgvector
 

### 2\. Async Embedding Pipeline

* Celery / Kafka for batch embedding jobs

 

### 3\. Multi-Tenant Prompt System

* org / team / engine isolation
* access control layer

 

### 4\. UI Dashboard

* Cluster visualization (UMAP / t-SNE)
* Duplicate inspection tool

 
### 5\. LLM-based Prompt Merging

* Suggest canonical merged templates
* Auto-deduplication suggestions

 

# Summary

This service provides a **clean, modular, production-ready foundation** for:

* Prompt deduplication
* Semantic search
* Embedding-based analytics

Designed for easy scaling and backend swapping.

