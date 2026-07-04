import uuid

from pydantic_validation import dump_model_json
from retriever import Retriever
from chunker import Chunker
from typing import Any
from llm import Llm
import json
import os


class RagSystem:
    def __init__(self) -> None:
        self.chunker = Chunker(vllm_path="./data/raw/vllm-0.10.1")
        self.retriever = Retriever()
        self.llm = Llm()

    def save_as_json(self, data: Any, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")

    def index(self, max_chunk_size: int = 2000) -> None:
        try:
            if 2000 < int(max_chunk_size) <= 0:
                raise ValueError(
                    "max_chunk_size must be a positive "
                    "integer and less than or equal to 2000."
                )
            self.chunker.chunk_size = max_chunk_size
            self.chunker.get_files_and_chunks()
            self.retriever.tokenize_chunks()
            print("Ingestion complete! Indices saved under data/processed/")
        except (ValueError, Exception) as e:
            print(f"Error occurred while indexing: {e}")
            return

    def search(self, query: str, k: int = 10) -> None:
        try:
            json = {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question": query,
                        "retrieved_sources": [
                            {
                                "file_path": source["file_path"],
                                "first_character_index": source[
                                    "first_character_index"
                                ],
                                "last_character_index": source[
                                    "last_character_index"
                                ],
                            }
                            for source in self.retriever.retrive(query, k)
                        ],
                    }
                ],
                "k": k,
            }
            self.save_as_json(
                json, "data/processed/output/search_results.json"
            )
        except Exception as e:
            print(f"Error in search: {e}")

    def answer(self, query: str, k: int = 10) -> None:
        try:
            rslt = self.llm.chat(
                prompt=query, context=self.retriever.retrive(query, k)
            )
            json = {
                "search_results": [
                    {
                        "question_id": "q1",
                        "question": query,
                        "retrieved_sources": [
                            {
                                "file_path": source["file_path"],
                                "first_character_index": source[
                                    "first_character_index"
                                ],
                                "last_character_index": source[
                                    "last_character_index"
                                ],
                            }
                            for source in rslt["retrieved_chunks"]
                        ],
                        "answer": rslt["answer"],
                    }
                ],
                "k": k,
            }
            self.save_as_json(
                json, "data/processed/output/answered_results.json"
            )
        except Exception as e:
            print(f"Error in answer: {e}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/processed/output/",
    ) -> None:
        try:
            search_results = self.retriever.retrieve_from_dataset(
                dataset_path, k
            )
            json = {
                "search_results": [
                    {
                        "question_id": str(uuid.uuid4()),
                        "question": item["prompt"],
                        "retrieved_sources": [
                            {
                                "file_path": source["file_path"],
                                "first_character_index": source[
                                    "first_character_index"
                                ],
                                "last_character_index": source[
                                    "last_character_index"
                                ],
                                "text": source["text"],
                            }
                            for source in item["retrieved_chunks"]
                        ],
                    }
                    for i, item in enumerate(search_results)
                ],
                "k": k,
            }
            self.save_as_json(
                json, f"{save_directory}/search_datasets_results.json"
            )
        except Exception as e:
            print(f"Error in search_dataset: {e}")

    def answer_dataset(
        self, student_search_result_path: str, save_directory: str
    ) -> None:
        try:
            answer_results = self.llm.answer_dataset(
                student_search_result_path=student_search_result_path,
                save_directory=save_directory,
            )
            json = {
                "search_results": [
                    {
                        "question_id": str(uuid.uuid4()),
                        "question": item["prompt"],
                        "retrieved_sources": [
                            {
                                "file_path": source["file_path"],
                                "first_character_index": source[
                                    "first_character_index"
                                ],
                                "last_character_index": source[
                                    "last_character_index"
                                ],
                            }
                            for source in item["retrieved_chunks"]
                        ],
                    }
                    for i, item in enumerate(answer_results)
                ],
                "k": answer_results["k"],
            }
            self.save_as_json(
                json, f"{save_directory}/answered_datasets_results.json"
            )
        except Exception as e:
            print(f"Error in answer_dataset: {e}")

    def evaluate(self) -> None:
        pass
