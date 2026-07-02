from retriever import Retriever
from chunker import Chunker
from llm import Llm

# import time


class RagSystem:
    def __init__(self) -> None:
        self.chunker = Chunker(vllm_path="./data/raw/vllm-0.10.1")
        self.retriever = Retriever()
        self.llm = Llm()

    def index(self, max_chunk_size: int = 2000) -> None:
        self.chunker.chunk_size = max_chunk_size
        # start_time = time.time()
        self.chunker.get_files_and_chunks()
        # end_time = time.time()
        # print(f"time taken for chuking: {end_time - start_time}")
        self.retriever.tokenize_chunks()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        print(f"Searching for: {query} with k={k}")
        self.retriever.retrieve(query, k)

    def answer(self, query: str, k: int = 10) -> None:
        print(f"Answering: {query} with k={k}")
        self.llm.chat(prompt=query, context=self.retriever.retrieve(query, k))

    def search_dataset(self, dataset_path: str, k: int = 10) -> None:
        pass

    def answer_dataset(
        self, student_search_result_path: str, save_directory: str
    ) -> None:
        pass
