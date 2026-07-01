from typing import Any
import bm25s
import json


class Retriever:
    def __init__(self) -> None:
        self.tokenized: Any = None
        self.retriver: Any = bm25s.BM25()

    def tokenize_chunks(self) -> None:
        try:
            with open("./data/processed/chunk/chunks.json", "r") as file:
                chunks = json.load(file)
            self.tokenized = bm25s.tokenize(
                [chunk["text"] for chunk in chunks], stopwords="en"
            )
            self.retriver.index(self.tokenized)
            self.retriver.save("data/processed/bm25_index")
        except Exception as e:
            print(f"Error in intokenize_chunks: {e}")

    def retrieve(self, prompt: str, k: int) -> Any:
        try:
            self.retriver = bm25s.BM25.load("data/processed/bm25_index")
            tokenized_query: Any = bm25s.tokenize(prompt)
            result, score = self.retriver(tokenized_query, k)
            print(result, score)
        except Exception as e:
            print(f"Error in retrieve: {e}")
