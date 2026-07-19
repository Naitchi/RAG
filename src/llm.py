from .pydantic_validation import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    load_student_search_results,
)
from tqdm import tqdm
import ollama


class Llm:
    """Generates grounded answers from retrieved context via Ollama."""

    def __init__(self) -> None:
        self.model = "qwen3:0.6b"

    @staticmethod
    def _build_prompt(question: str, texts: list[str]) -> str:
        """Build the context-grounded prompt sent to the model."""
        formatted_context = "\n\n".join(
            f"Document {i+1}:\n{text}" for i, text in enumerate(texts)
        )
        return f"""Answer the question based on the following context.
        <context>
        {formatted_context}
        </context>
        <question>
        {question}
        </question>"""

    def chat(self, question: str, texts: list[str]) -> str:
        """Ask the model to answer `question` grounded in `texts`.

        Args:
            question: The user's question.
            texts: Retrieved source texts to use as context.

        Returns:
            The model's answer.

        Raises:
            Any exception from the Ollama call, rather than returning an
            empty string that would look like a legitimate (if useless)
            answer to the caller.
        """
        m = self._build_prompt(question, texts)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": m}],
            options={"num_ctx": 4098, "temperature": 0.2},
        )
        answer = str(response["message"]["content"])
        print(f"Response: {answer}")
        return answer

    def answer_dataset(
        self, student_search_results_path: str
    ) -> StudentSearchResultsAndAnswer:
        """Generate an answer for every question in a search-results file.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file (typically the output of `search_dataset`).

        Returns:
            A `StudentSearchResultsAndAnswer` with one generated answer
            per question.

        Raises:
            Any exception from loading `student_search_results_path`,
            so a broken/missing input file is never mistaken for a
            valid empty result. A single question's generation failing,
            however, does not abort the rest of the batch: it is caught
            locally and recorded with an empty answer.
        """
        dataset: StudentSearchResults = load_student_search_results(
            student_search_results_path
        )
        answered_results: list[MinimalAnswer] = []
        for item in tqdm(dataset.search_results, desc="Answering dataset"):
            texts = [
                source.text
                for source in item.retrieved_sources
                if source.text
            ]
            try:
                answer = self.chat(question=item.question, texts=texts)
            except Exception as e:
                print(f"Error answering question {item.question_id}: {e}")
                answer = ""
            answered_results.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer,
                )
            )
        return StudentSearchResultsAndAnswer(
            search_results=answered_results,
            k=dataset.k,
        )
