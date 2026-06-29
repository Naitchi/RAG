from retriever import Retriever
from chunker import Chunker

# import time


# start_time = time.time()
# chunker.get_files_and_chunks()
# end_time = time.time()
# print(f"time taken for chuking: {end_time - start_time}")


class RagSystem:
    def __init__(self) -> None:
        self.chunker = Chunker(vllm_path="./vllm-0.10.1")
        self.retriever = Retriever()

    # TODO The maximum chunk size is 2000 characters and it has to be
    # configurable through a CLI argument
    def index(self, max_chunk_size: int = 2000) -> None:
        pass

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
