# TODO - ft_rag

## Phase 1 - Project Setup

* [x] Initialize project with `uv`
* [ ] Configure `pyproject.toml`

* [ ] Install dependencies:
  * [ ] pydantic
  * [ ] fire
  * [ ] tqdm
  * [ ] bm25s
  * [ ] transformers
  * [ ] torch
  * [ ] numpy
  
* [ ] Configure flake8
* [x] Verify Python 3.10 compatibility

### Project Structure

* [ ] Create source tree

```text
src/student/

├── __main__.py
├── cli.py
├── models.py
├── config.py

├── ingestion/
├── retrieval/
├── generation/
├── evaluation/
└── storage/
```

---

# Phase 2 - Pydantic Models

* [ ] Implement `MinimalSource`
* [ ] Implement `UnansweredQuestion`
* [ ] Implement `AnsweredQuestion`
* [ ] Implement `RagDataset`
* [ ] Implement `MinimalSearchResults`
* [ ] Implement `MinimalAnswer`
* [ ] Implement `StudentSearchResults`
* [ ] Implement `StudentSearchResultsAndAnswer`

---

# Phase 3 - Repository Ingestion

## File Discovery

* [ ] Recursively scan repository
* [ ] Ignore:

  * [ ] `.git`
  * [ ] `.venv`
  * [ ] `__pycache__`
  * [ ] binary files

## Supported Files

* [ ] `.py`
* [ ] `.md`
* [ ] `.rst`
* [ ] `.txt`

---

# Phase 4 - Chunking

## Python Chunking

* [ ] Parse Python files using AST
* [ ] Chunk by class
* [ ] Chunk by function
* [ ] Chunk by method
* [ ] Preserve character offsets

## Documentation Chunking

* [ ] Split by markdown headers
* [ ] Split by sections
* [ ] Respect maximum chunk size
* [ ] Make chunk size configurable

Default:

```bash
--max_chunk_size 2000
```

---

# Phase 5 - BM25 Indexing

## Preprocessing

* [ ] Normalize text
* [ ] Tokenize content
* [ ] Lowercase text

## Index Creation

* [ ] Build BM25 corpus
* [ ] Create BM25 index
* [ ] Save index to disk

Output:

```text
data/processed/

├── bm25_index/
└── chunks/
```

---

# Phase 6 - Storage

* [ ] Save chunks metadata
* [ ] Save BM25 index
* [ ] Save index metadata
* [ ] Implement loading utilities

Each chunk must contain:

* [ ] file_path
* [ ] first_character_index
* [ ] last_character_index
* [ ] content

---

# Phase 7 - Search Command

Command:

```bash
uv run python -m student search "query"
```

Tasks:

* [ ] Load index
* [ ] Execute BM25 retrieval
* [ ] Return top-k chunks
* [ ] Convert results to `MinimalSource`

---

# Phase 8 - Dataset Search

Command:

```bash
uv run python -m student search_dataset
```

Tasks:

* [ ] Load dataset
* [ ] Iterate over questions
* [ ] Retrieve top-k results
* [ ] Display progress bar with tqdm
* [ ] Save JSON output

---

# Phase 9 - Qwen Integration

Model:

```text
Qwen/Qwen3-0.6B
```

Tasks:

* [ ] Load tokenizer
* [ ] Load model
* [ ] Create inference wrapper
* [ ] Handle loading errors

---

# Phase 10 - Answer Generation

Command:

```bash
uv run python -m student answer "question"
```

Pipeline:

```text
Question
↓
Retrieve top-k chunks
↓
Build prompt
↓
Qwen
↓
Answer
```

Tasks:

* [ ] Build RAG prompt
* [ ] Limit context size
* [ ] Generate answer
* [ ] Include source references

---

# Phase 11 - Dataset Answering

Command:

```bash
uv run python -m student answer_dataset
```

Tasks:

* [ ] Load search results
* [ ] Generate answers
* [ ] Save output JSON
* [ ] Display tqdm progress

---

# Phase 12 - Evaluation

## Recall@K

* [ ] Implement overlap calculation
* [ ] Count source as found if overlap >= 5%
* [ ] Compute Recall@1
* [ ] Compute Recall@3
* [ ] Compute Recall@5
* [ ] Compute Recall@10

Output example:

```text
Recall@1: 0.45
Recall@3: 0.59
Recall@5: 0.65
Recall@10: 0.72
```

---

# Phase 13 - CLI

Required commands:

* [ ] index
* [ ] search
* [ ] search_dataset
* [ ] answer
* [ ] answer_dataset
* [ ] evaluate

Implementation:

* [ ] Python Fire CLI

---

# Phase 14 - Error Handling

* [ ] Handle file reading errors
* [ ] Handle JSON parsing errors
* [ ] Handle AST parsing errors
* [ ] Handle model loading errors
* [ ] Handle retrieval errors
* [ ] Ensure no unhandled exceptions

---

# Phase 15 - Performance Targets

## Mandatory

* [ ] Indexing under 5 minutes
* [ ] Cold start under 60 seconds
* [ ] 1000 queries under 90 seconds
* [ ] Recall@5 ≥ 80% docs
* [ ] Recall@5 ≥ 50% code

## Optimization

* [ ] Improve chunking strategy
* [ ] Separate code/doc indexes
* [ ] Experiment with reranking
* [ ] Experiment with hybrid retrieval

---

# README

* [ ] Project description
* [ ] Installation instructions
* [ ] Usage examples
* [ ] System architecture
* [ ] Chunking strategy
* [ ] Retrieval method
* [ ] Performance analysis
* [ ] Design decisions
* [ ] Challenges faced
* [ ] AI usage disclosure

---

# MVP Priority Order

1. [ ] Pydantic models
2. [ ] File ingestion
3. [ ] Chunking
4. [ ] BM25 indexing
5. [ ] Search
6. [ ] Dataset search
7. [ ] Recall evaluation
8. [ ] Qwen integration
9. [ ] Answer generation
10. [ ] README
