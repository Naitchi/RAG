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
            The model's answer, or an empty string if generation failed.
        """
        try:
            m = self._build_prompt(question, texts)
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": m}],
                options={"num_ctx": 4098, "temperature": 0.2},
            )
            answer = str(response["message"]["content"])
            print(f"Response: {answer}")
            return answer
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""

    def answer_dataset(
        self, student_search_results_path: str
    ) -> StudentSearchResultsAndAnswer:
        """Generate an answer for every question in a search-results file.

        Args:
            student_search_results_path: Path to a `StudentSearchResults`
                JSON file (typically the output of `search_dataset`).

        Returns:
            A `StudentSearchResultsAndAnswer` with one generated answer
            per question. Returns an empty result set on failure, rather
            than raising, so a single bad file never crashes the CLI.
        """
        try:
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
                answer = self.chat(question=item.question, texts=texts)
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
        except Exception as e:
            print(f"Error in answer_dataset: {e}")
            return StudentSearchResultsAndAnswer(search_results=[], k=0)
