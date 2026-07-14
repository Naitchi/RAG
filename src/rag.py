from pydantic import BaseModel
from typing import Any
import uuid
import json
import os

from .evaluation import Evaluation
from .retriever import Retriever
from .chunker import Chunker
from .llm import Llm


class RagSystem:
    """CLI-facing orchestrator wiring together chunking, retrieval,
    answer generation and evaluation. Each public method is exposed as a
    `uv run python -m src <command>` CLI command via Python Fire.
    """

    def __init__(self) -> None:
        self.chunker = Chunker(vllm_path="./data/raw/vllm-0.10.1")
        self.retriever = Retriever()
        self.llm = Llm()
        self.evaluater = Evaluation()

    def save_as_json(self, data: Any, file_path: str) -> None:
        """Write `data` (a pydantic model or plain JSON-able value) to
        `file_path`, creating parent directories as needed.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                if isinstance(data, BaseModel):
                    json.dump(data.model_dump(), f, indent=4)
                else:
                    json.dump(data, f, indent=4)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to {file_path}: {e}")
            return

    def index(self, max_chunk_size: str = "2000") -> None:
        """Chunk `data/raw/` and build the BM25 index under `data/processed/`.

        Args:
            max_chunk_size: Maximum chunk size in characters (1-2000).
        """
        try:
            if not 0 < int(max_chunk_size) <= 2000:
                raise ValueError(
                    "max_chunk_size must be a positive "
                    "integer and less than or equal to 2000."
                )
            self.chunker.chunk_size = int(max_chunk_size)
            self.chunker.get_files_and_chunks()
            self.retriever.tokenize_chunks()
            print("Ingestion complete! Indices saved under data/processed/")
        except (ValueError, Exception) as e:
            print(f"Error occurred while indexing: {e}")
            return

    def search(self, query: str, k: int = 10) -> None:
        """Print the top-k retrieved sources for a single query.

        Args:
            query: The natural-language question to search for.
            k: Number of sources to retrieve.
        """
        try:
            if not query or not query.strip():
                raise ValueError("query must not be empty")
            if k <= 0:
                raise ValueError("k must be a positive integer")
            results = self.retriever.retrive(query, k)
            if not results:
                print("No results found.")
                return
            for source in results:
                print(
                    f"{source['file_path']} "
                    f"[{source['first_character_index']}:"
                    f"{source['last_character_index']}]"
                )
        except Exception as e:
            print(f"Error in search: {e}")
            return

    def answer(self, query: str, k: int = 10) -> None:
        """Print a generated answer and its sources for a single query.

        Args:
            query: The natural-language question to answer.
            k: Number of sources to retrieve as context.
        """
        try:
            if not query or not query.strip():
                raise ValueError("query must not be empty")
            if k <= 0:
                raise ValueError("k must be a positive integer")
            context = self.retriever.retrive(query, k)
            if not context:
                print("No relevant context found for this query.")
                return
            texts = [source["text"] for source in context]
            answer_text = self.llm.chat(question=query, texts=texts)
            print(f"\nAnswer:\n{answer_text}\n")
            print("Sources:")
            for source in context:
                print(
                    f"  {source['file_path']} "
                    f"[{source['first_character_index']}:"
                    f"{source['last_character_index']}]"
                )
        except Exception as e:
            print(f"Error in answer: {e}")
            return

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
    ) -> None:
        """Run `search` over every question in a dataset and save the result.

        Args:
            dataset_path: Path to a `RagDataset` JSON file to search over.
            save_directory: Directory to write the `StudentSearchResults`
                JSON file to, under the input dataset's own file name.
            k: Number of sources to retrieve per question.
        """
        try:
            if not save_directory or not save_directory.strip():
                raise ValueError("save_directory must not be empty")
            if 0 >= k or k > 10:
                raise ValueError(
                    "k must be a positive integer between 1 and 10"
                )
            search_results = self.retriever.retrieve_from_dataset(
                dataset_path, k
            )
            json_chunk: dict[str, Any] = {
                "search_results": [
                    {
                        "question_id": item.get(
                            "question_id",
                            str(uuid.uuid4()),
                        ),
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
                    for item in search_results
                ],
                "k": k,
            }
            self.save_as_json(
                json_chunk,
                os.path.join(save_directory, os.path.basename(dataset_path)),
            )
        except Exception as e:
            print(f"Error in search_dataset: {e}")
            return

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate an answer for every question in a search-results file.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file, typically produced by `search_dataset`.
            save_directory: Directory to write the resulting
                `StudentSearchResultsAndAnswer` JSON file to, under the
                input file's own name.
        """
        try:

            answer_results = self.llm.answer_dataset(
                student_search_results_path=student_search_results_path,
            )
            self.save_as_json(
                answer_results,
                os.path.join(
                    save_directory,
                    os.path.basename(student_search_results_path),
                ),
            )
        except Exception as e:
            print(f"Error in answer_dataset: {e}")
            return

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> None:
        """Print recall@k for a student search-results file, for local use.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file (typically the output of `search_dataset`).
            dataset_path: Path to the matching `AnsweredQuestions`
                ground-truth JSON file.
            k: Highest k to report recall for.
            max_context_length: Maximum allowed source length in
                characters.
        """
        try:
            self.evaluater.evaluate_dataset(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path,
                k=k,
                max_context_length=max_context_length,
            )
        except Exception as e:
            print(f"Error in evaluate: {e}")
            return
