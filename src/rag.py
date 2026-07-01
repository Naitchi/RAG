from retriever import Retriever
from chunker import Chunker

import time


class RagSystem:
    def __init__(self) -> None:
        self.chunker = Chunker(vllm_path="./vllm-0.10.1")
        self.retriever = Retriever()

    def index(self, max_chunk_size: int = 2000) -> None:
        self.chunker.chunk_size = max_chunk_size
        start_time = time.time()
        self.chunker.get_files_and_chunks()
        end_time = time.time()
        print(f"time taken for chuking: {end_time - start_time}")
        self.retriever.tokenize_chunks()

    def search(self, query: str, k: int = 10) -> None:
        pass

    def answer(self, query: str, k: int = 10) -> None:
        pass

    def search_dataset(self, dataset_path: str, k: int = 10) -> None:
        pass

    def answer_dataset(
        self, student_search_result_path: str, save_directory: str
    ) -> None:
        pass
