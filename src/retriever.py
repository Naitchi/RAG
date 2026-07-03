from typing import Any
import bm25s
import json


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
            self.retriver.save(self.index_path, corpus=texts)
        except Exception as e:
            print(f"Error in intokenize_chunks: {e}")

    def retrieve(self, prompt: str, k: int) -> Any:
        try:
            self.retriver = bm25s.BM25.load(
                "data/processed/bm25_index", load_corpus=True
            )
            tokenized_query: Any = bm25s.tokenize(prompt)
            result, score = self.retriver.retrieve(tokenized_query, k=k)
            print(f"Retrieved {len(result)} chunks with scores: {score}")
            return result[0].tolist()
        except Exception as e:
            print(f"Error in retrieve: {e}")

    def retrieve_from_dataset(self, dataset_path: str, k: int) -> Any:
        try:
            self.retriver = bm25s.BM25.load(self.index_path, load_corpus=True)
            with open(dataset_path, "r") as file:
                dataset = json.load(file)
            results: list[dict[str, Any]] = []
            for item in dataset:
                prompt = item.get("prompt", "")
                tokenized_query: Any = bm25s.tokenize(prompt)
                result, score = self.retriver.retrieve(tokenized_query, k=k)
                results.append(
                    {
                        "prompt": prompt,
                        "retrieved_chunks": [r.tolist() for r in result],
                        "scores": score.tolist(),
                    }
                )
            return results
        except Exception as e:
            print(f"Error in retrieve_from_dataset: {e}")
