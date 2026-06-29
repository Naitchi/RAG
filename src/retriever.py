from typing import Any

import bm25s  # type: ignore


class Retriever:
    def __init__(self) -> None:
        self.tokenized: Any = None
        self.retriver: Any = bm25s.BM25()

    def tokenize_chunks(self, chunks: list[dict[str, str]]) -> None:
        self.tokenized: Any = bm25s.tokenize(  # type: ignore
            [chunk["chunk"] for chunk in chunks], stopwords="en"
        )
        self.retriver.index(self.tokenized)

    def retrieve(self, prompt: str, k: int) -> Any:
        if self.tokenized is None:
            print("Erreur il me faut le tokenize des chunks")
            return

        tokenized_query: Any = bm25s.tokenize(prompt)  # type: ignore
        result, score = self.retriver(tokenized_query, k)
        print(result, score)
