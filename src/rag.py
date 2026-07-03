from retriever import Retriever
from chunker import Chunker
from llm import Llm


class RagSystem:
    def __init__(self) -> None:
        self.chunker = Chunker(vllm_path="./data/raw/vllm-0.10.1")
        self.retriever = Retriever()
        self.llm = Llm()

    def index(self, max_chunk_size: int = 2000) -> None:
        self.chunker.chunk_size = max_chunk_size
        self.chunker.get_files_and_chunks()
        self.retriever.tokenize_chunks()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        print(f"Searching for: {query} with k={k}")
        self.retriever.retrieve(query, k)

    def answer(self, query: str, k: int = 10) -> None:
        print(f"Answering: {query} with k={k}")
        self.llm.chat(prompt=query, context=self.retriever.retrieve(query, k))

    def search_dataset(self, dataset_path: str, k: int = 10) -> None:
        print(f"Searching dataset: {dataset_path} with k={k}")
        print(
            f"Search complete! "
            f"{self.retriever.retrieve_from_dataset(dataset_path, k)}"
        )

    def answer_dataset(
        self, student_search_result_path: str, save_directory: str
    ) -> None:
        print(
            f"Answering dataset: {student_search_result_path} "
            f"with save directory: {save_directory}"
        )
        self.llm.answer_dataset(
            student_search_result_path=student_search_result_path,
            save_directory=save_directory,
        )
