from typing import List
from .pydantic_validation import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluation:
    """Computes recall@k for a student's retrieval output, for local use."""

    def check_chunk_length(
        self, chunk_size: int, chunks: list[MinimalSource]
    ) -> bool:
        """Return False if any source spans more than `chunk_size` chars."""
        for chunk in chunks:
            if (
                chunk.last_character_index - chunk.first_character_index
                > chunk_size
            ):
                return False
        return True

    def check_number_answers_with_sources(
        self, list_questions: list, student: bool = False
    ) -> int:
        """Count questions that have at least one attached source.

        Args:
            list_questions: Either `AnsweredQuestion` (ground truth) or
                `MinimalSearchResults` (student output) items, depending
                on `student`.
            student: If True, count `retrieved_sources` on student
                results; otherwise count `sources` on ground-truth
                `AnsweredQuestion` items.

        Returns:
            The number of questions with at least one source.
        """
        count = 0
        if student:
            for question in list_questions:
                if len(question.retrieved_sources) > 0:
                    count += 1
            return count
        for question in list_questions:
            if (
                isinstance(question, AnsweredQuestion)
                and len(question.sources) > 0
            ):
                count += 1
        return count

    def calculate_recall(
        self,
        student_answers: StudentSearchResults,
        dataset: RagDataset,
        k: int,
    ) -> float:
        """Compute recall@k, averaged over every matched question.

        For each question, recall is the share of its ground-truth
        sources found (via `_has_overlap`) among the student's top-k
        retrieved sources. Questions are matched between `student_answers`
        and `dataset` by their `question` text.

        Args:
            student_answers: The student's retrieval output.
            dataset: The ground-truth dataset (`AnsweredQuestion` entries).
            k: Number of top retrieved sources to consider per question.

        Returns:
            The mean per-question recall, rounded to 3 decimals, or 0.0
            if no question could be matched.
        """
        correct_sources: dict[str, List] = {
            q.question: q.sources
            for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion)
        }

        total_recall = 0.0
        evaluated = 0

        for result in student_answers.search_results:
            if result.question not in correct_sources:
                continue

            correct = correct_sources[result.question]
            retrieved = result.retrieved_sources[:k]

            found = 0
            for correct_source in correct:
                for retrieved_source in retrieved:
                    if self._has_overlap(correct_source, retrieved_source):
                        found += 1
                        break

            total_recall += found / len(correct) if correct else 0.0
            evaluated += 1

        return round(total_recall / evaluated, 3) if evaluated > 0 else 0.0

    def _has_overlap(
        self,
        correct: MinimalSource,
        retrieved: MinimalSource,
        threshold: float = 0.05,
    ) -> bool:
        """Return True if `retrieved` counts as a match for `correct`.

        A match requires the same `file_path` and a character-range
        overlap of at least `threshold` of the ground-truth span's
        length (IoU-style, relative to `correct`'s length only).
        """

        if correct.file_path != retrieved.file_path:
            return False

        overlap_start = max(
            correct.first_character_index, retrieved.first_character_index
        )
        overlap_end = min(
            correct.last_character_index, retrieved.last_character_index
        )
        overlap = max(0, overlap_end - overlap_start)

        correct_len = (
            correct.last_character_index - correct.first_character_index
        )
        if correct_len == 0:
            return False

        return bool((overlap / correct_len) >= threshold)

    def evaluate_dataset(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int,
        max_context_length: int,
    ) -> None:
        """Load a student output and a ground-truth dataset and print recall@k.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file (typically the output of `search_dataset`).
            dataset_path: Path to the matching `AnsweredQuestions`
                ground-truth JSON file.
            k: Highest k to report; recall is printed for every value in
                [1, 3, 5, 10] up to `k`.
            max_context_length: Maximum allowed source length in
                characters, used to validate the student output.

        Raises:
            Any exception from loading either file or from an invalid
            `k`, rather than printing an error and returning silently:
            the caller (`RagSystem.evaluate`) already has its own
            try/except and must be the one deciding how to report it.
        """
        with open(student_search_results_path, "r") as f:
            student_answers = StudentSearchResults.model_validate_json(
                f.read()
            )

        with open(dataset_path, "r") as f:
            dataset = RagDataset.model_validate_json(f.read())
        if k not in [1, 3, 5, 10]:
            raise ValueError("k must be one of [1, 3, 5, 10]")
        chunks = [
            source
            for item in student_answers.search_results
            for source in item.retrieved_sources
        ]

        print(
            "Student data is valid: "
            f"{self.check_chunk_length(max_context_length, chunks)}"
        )
        print(f"Total number of questions: {len(dataset.rag_questions)}")
        print(
            "Total number of questions with sources: "
            f"{self.check_number_answers_with_sources(
                dataset.rag_questions)}"
        )
        print(
            "Total number of questions with student sources: "
            f"{self.check_number_answers_with_sources(
                student_answers.search_results, True)}"
        )

        print("\nEvaluation Results")
        print("========================================")
        print(f"Questions evaluated: {len(student_answers.search_results)}")
        print(
            "Recall@1: "
            f"{self.calculate_recall(student_answers, dataset, k=1)}"
        )
        if k > 1:
            print(
                "Recall@3: "
                f"{self.calculate_recall(student_answers, dataset, k=3)}"
            )
        if k > 3:
            print(
                "Recall@5: "
                f"{self.calculate_recall(student_answers, dataset, k=5)}"
            )
        if k > 5:
            print(
                "Recall@10: "
                f"{self.calculate_recall(student_answers, dataset, k=10)}"
            )
