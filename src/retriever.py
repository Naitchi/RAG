from typing import Any
import bm25s
import json

from pydantic_validation import (
    UnansweredQuestion,
    load_rag_dataset,
)


class Retriever:
    def __init__(self) -> None:
        self.tokenized: Any = None
        self.retriver: Any = bm25s.BM25()
        self.index_path: str = "data/processed/bm25_index"

    def tokenize_chunks(self) -> None:
        try:
            with open("./data/processed/chunk/chunks.json", "r") as file:
                chunks = json.load(file)
            texts = [chunk["text"] for chunk in chunks]
            self.tokenized = bm25s.tokenize(texts, stopwords="en")
            self.retriver.index(self.tokenized)
            self.retriver.save(self.index_path, corpus=chunks)
        except Exception as e:
            print(f"Error in intokenize_chunks: {e}")

    def retrive(self, prompt: str, k: int) -> list[dict[str, Any]]:
        try:
            self.retriver = bm25s.BM25.load(self.index_path, load_corpus=True)
            tokenized_query: Any = bm25s.tokenize(prompt)
            result, _ = self.retriver.retrieve(tokenized_query, k=k)
            return result[0].tolist()
        except Exception as e:
            print(f"Error in retrieve: {e}")

    def retrieve_from_dataset(
        self, dataset_path: str, k: int
    ) -> list[dict[str, Any]]:
        try:
            self.retriver = bm25s.BM25.load(self.index_path, load_corpus=True)
            prompt_dataset: list[str] = [
                q.question
                for q in load_rag_dataset(dataset_path).rag_questions
            ]
            data: list[dict[str, Any]] = []
            tokenized_query: Any = bm25s.tokenize(prompt_dataset)
            results_retrieve, _ = self.retriver.retrieve(tokenized_query, k=k)
            for result, prompt in zip(results_retrieve, prompt_dataset):
                data.append(
                    {
                        "prompt": prompt,
                        "retrieved_chunks": result.tolist(),
                    }
                )
            return data
        except Exception as e:
            print(f"Error in retrieve_from_dataset: {e}")
