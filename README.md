*This project has been created as part of the 42 curriculum by bclairot.*

# RAG

## Description

This project implements a Retrieval-Augmented Generation pipeline over a local snapshot of vLLM 0.10.1. The goal is to ingest documentation and source code, split it into searchable chunks, build a BM25 index, retrieve the most relevant passages for a question, and optionally generate an answer with a local Ollama model.

The repository is structured around a fully local workflow:

- raw material lives in `data/raw/vllm-0.10.1`
- processed chunks and indexes are stored in `data/processed`
- datasets for evaluation are stored in `data/datasets`

## Instructions

### Requirements

- Python 3.10 or newer
- `uv`
- Ollama installed locally
- the `qwen3:0.6b` model pulled in Ollama

### Installation

```bash
make install
```

This installs the Python dependencies, starts the Ollama installation flow, and pulls `qwen3:0.6b`.

If you prefer manual setup:

```bash
uv sync
ollama pull qwen3:0.6b
```

### Execution

The Makefile starts Ollama and forwards arguments to the CLI:

```bash
make run ARGS="index"
```

You can also run the package directly:

```bash
uv run python -m src index
```

## System architecture

The RAG pipeline is made of four main components:

- `Chunker`: reads the vLLM source tree and splits files into smaller searchable chunks
- `Retriever`: loads the BM25 index and finds the most relevant chunks for a user question
- `LLM`: sends the retrieved chunks to Ollama and generates the final answer
- `Evaluation`: compares retrieved chunks with the reference datasets and computes recall@k

They interact in this order:

1. The chunker scans `data/raw/vllm-0.10.1` and writes chunk metadata to `data/processed/chunk/chunks.json`.
2. The retriever builds a BM25 index from those chunks and uses it to search for relevant passages.
3. The LLM receives the top chunks as context and produces an answer.
4. The evaluation step reuses the same retrieval output to measure recall@k on the public datasets.

## Chunking Strategy

The chunker uses different strategies depending on the file type:

- plain text and reStructuredText files are segmented with `RecursiveCharacterTextSplitter`
- Markdown files use heading-aware separators so sections stay semantically meaningful
- Python files are parsed with `ast`, and each function or class definition becomes a candidate chunk

Implementation details:

- default chunk size is 2000 characters
- chunk overlap is 10 percent of the base chunk size
- the splitter uses smaller working chunks internally (`chunk_size = 75 percent of the base size`)
- chunks shorter than 10 percent of the base size are discarded
- whitespace and punctuation spacing are normalized before saving

Each chunk keeps its source file path and character range so retrieval results can be traced back to the original document.

## Retrieval Method

Retrieval is based on BM25 through the `bm25s` library.

- chunk texts are tokenized with English stop words
- the index is saved to `data/processed/bm25_index`
- queries are tokenized with the same BM25 pipeline
- results are ranked by BM25 score and the top `k` chunks are returned

This choice keeps the system lightweight, fast, and easy to reproduce without requiring embeddings or a vector database.

## Performance Analysis

Recall@k was computed on the public answered datasets using the current index in `data/processed/bm25_index`.

### Code dataset, 100 questions
<!-- TODO  mettre les vrai valeurs --> 
- Recall@1: 0.020
- Recall@3: 0.050
- Recall@5: 0.070
- Recall@10: 0.080

### Documentation dataset, 100 questions

- Recall@1: 0.630
- Recall@3: 0.760
- Recall@5: 0.840
- Recall@10: 0.870

The system performs much better on documentation than on code. That gap is expected: code questions often depend on symbolic names, implementation details, or local context that lexical BM25 matching does not always surface at rank 1.

## Design Decisions

- BM25 was chosen instead of dense embeddings to keep indexing simple, explainable, and fast on a local corpus.
- AST-based chunking for Python files preserves code structure better than naive fixed-width splitting.
- Source offsets are stored with each chunk so retrieved passages can be audited against the original file.
- Pydantic models validate dataset and output formats before evaluation.
- Fire provides a small CLI surface without adding a custom argument parser.
- Ollama keeps generation local and avoids any external API dependency.

## Challenges Faced

- The project subject was really dense and unclear, which made it hard to understand what was expected
- Code chunks are harder to evaluate than prose because exact source overlap is sensitive to chunk boundaries.
- Some questions map to multiple possible code regions, which makes sparse retrieval more brittle.
- The pipeline must keep file paths and character offsets consistent across chunking, indexing, and evaluation.
- Mixing prose and code required different chunking strategies to avoid losing semantic structure.

The main fix was to preserve metadata at chunk creation time and reuse the same chunk records during retrieval and evaluation.

## Example Usage

### Build the index

```bash
make run ARGS="index"
```

### Search for relevant chunks

```bash
make run ARGS="search --query='What is the default value of trust_remote_code in vLLM's LLM class constructor?' --k=5"
```

### Generate an answer

```bash
make run ARGS="answer --query='What hardware platforms does vLLM support?' --k=5"
```

### Evaluate retrieval

```bash
make run ARGS="evaluate --student_answer_path=data/processed/output/answered_datasets_results.json --dataset_path=data/datasets/AnsweredQuestions/dataset_docs_public.json --k=10 --max_context_length=2000"
```

### Process a dataset

```bash
make run ARGS="search_dataset data/datasets/UnansweredQuestions/dataset_docs_public.json 10 data/processed/output"
```

## Resources

### References

- vLLM documentation: https://docs.vllm.ai/
- BM25 background: https://en.wikipedia.org/wiki/Okapi_BM25
- bm25s project: https://github.com/xhluca/bm25s
- LangChain text splitters: https://python.langchain.com/docs/concepts/text_splitters/
- Ollama documentation: https://ollama.com/
- Pydantic documentation: https://docs.pydantic.dev/

### AI Usage

AI was used to help write a first draft and structure this README, and to help with documentation on the libraries.
