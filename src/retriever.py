from .pydantic_validation import load_rag_dataset
from typing import Any
import bm25s
import json


class Retriever:
    """BM25-based lexical retrieval over the indexed chunk corpus."""

    def __init__(self) -> None:
        self.tokenized: Any = None
        self.retriver: Any = bm25s.BM25(method="robertson", k1=1.5, b=0.5)
        self.index_path: str = "data/processed/bm25_index"

    def tokenize_chunks(self) -> None:
        """Tokenize `data/processed/chunk/chunks.json` and build/save the
        BM25 index under `self.index_path`.

        Raises on failure instead of swallowing it, so a broken index
        build is never mistaken for a successful one by the caller.
        """
        with open("./data/processed/chunk/chunks.json", "r") as file:
            chunks = json.load(file)
        texts = [chunk["text"] for chunk in chunks]
        self.tokenized = bm25s.tokenize(texts, stopwords="en")
        self.retriver.index(self.tokenized)
        self.retriver.save(self.index_path, corpus=chunks)

    def retrive(self, prompt: str, k: int) -> list[dict[str, Any]]:
        """Return the top-k chunks for a single query.

        Args:
            prompt: The natural-language query.
            k: Number of chunks to retrieve.

        Returns:
            The top-k chunk records (each with `file_path`,
            `first_character_index`, `last_character_index`, `text`).

        Raises:
            Any exception from loading the index or querying it, rather
            than returning an empty list that would look like a
            legitimate zero-result search.
        """
        self.retriver = bm25s.BM25.load(self.index_path, load_corpus=True)
        tokenized_query: Any = bm25s.tokenize(prompt)
        result, _ = self.retriver.retrieve(tokenized_query, k=k)
        return result[0].tolist()

    def retrieve_from_dataset(
        self, dataset_path: str, k: int
    ) -> list[dict[str, Any]]:
        """Run `retrive` for every question in a dataset.

        Args:
            dataset_path: Path to a `RagDataset` JSON file.
            k: Number of chunks to retrieve per question.

        Returns:
            One entry per question, each with `question_id`, `prompt`
            and `retrieved_chunks`.

        Raises:
            Any exception from loading the dataset, the index, or
            querying it, rather than returning an empty list that a
            caller could mistake for "no matches" and save as if it
            were a valid result.
        """
        self.retriver = bm25s.BM25.load(self.index_path, load_corpus=True)
        questions = load_rag_dataset(dataset_path).rag_questions
        prompt_dataset: list[str] = [q.question for q in questions]
        data: list[dict[str, Any]] = []
        tokenized_query: Any = bm25s.tokenize(prompt_dataset)
        results_retrieve, _ = self.retriver.retrieve(tokenized_query, k=k)
        for result, question in zip(results_retrieve, questions):
            data.append(
                {
                    "question_id": question.question_id,
                    "prompt": question.question,
                    "retrieved_chunks": result.tolist(),
                }
            )
        return data
