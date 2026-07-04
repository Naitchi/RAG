from typing import Any

from pydantic_validation import MinimalSource


class Evaluation:

    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_overlap(
        source1: dict[str, Any], source2: dict[str, Any]
    ) -> float:
        """Calculate character overlap between two sources."""
        start1 = source1["first_character_index"]
        end1 = source1["last_character_index"]
        start2 = source2["first_character_index"]
        end2 = source2["last_character_index"]

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap = max(0, overlap_end - overlap_start)

        source2_length = end2 - start2
        if source2_length == 0:
            return 0.0
        return overlap / source2_length

    def is_source_found(
        self,
        retrieved: dict[str, Any],
        correct: dict[str, Any],
        threshold: float = 0.05,
    ) -> bool:
        """Check if retrieved source overlaps with correct source by threshold."""
        if retrieved["file_path"] != correct["file_path"]:
            return False
        return self.calculate_overlap(retrieved, correct) >= threshold

    def recall_at_k(
        self,
        retrieved_sources: list[dict[str, Any] | MinimalSource],
        correct_sources: list[dict[str, Any] | MinimalSource],
    ) -> float:
        """
        Calculate recall@k: fraction of correct sources found in retrieved.

        A source is found if it has >= 5% overlap with any correct source.
        """
        if not correct_sources:
            return 1.0

        retrieved_dicts = [
            s.model_dump() if isinstance(s, MinimalSource) else s
            for s in retrieved_sources
        ]
        correct_dicts = [
            s.model_dump() if isinstance(s, MinimalSource) else s
            for s in correct_sources
        ]

        found_count = 0
        for correct in correct_dicts:
            for retrieved in retrieved_dicts:
                if self.is_source_found(retrieved, correct):
                    found_count += 1
                    break

        return found_count / len(correct_dicts)

    def evaluate_dataset(
        self,
        dataset: list[dict[str, Any]],
        k_values: list[int] = [1, 3, 5, 10],
    ) -> dict[str, float]:
        """
        Evaluate dataset and compute recall@k for each k.
        Dataset items must have:
        - sources: list of correct sources
        - retrieved_sources: list of retrieved sources
        """
        results = {f"recall@{k}": 0.0 for k in k_values}
        count = 0

        for item in dataset:
            correct = item.get("sources", [])
            if not correct:
                continue

            for k in k_values:
                retrieved = item.get("retrieved_sources", [])[:k]
                score = self.recall_at_k(retrieved, correct)
                results[f"recall@{k}"] += score
            count += 1

        if count > 0:
            for key in results:
                results[key] /= count

        return results
