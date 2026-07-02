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
            texts = [chunk["text"] for chunk in chunks]
            self.tokenized = bm25s.tokenize(texts, stopwords="en")
            self.retriver.index(self.tokenized)
            self.retriver.save("data/processed/bm25_index", corpus=texts)
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
