from pydantic_validation import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    load_student_search_results,
)
from typing import Any
import ollama


class Llm:
    def __init__(self) -> None:
        self.model = "qwen3:0.6b"

    def chat(self, prompt: str, context: list[dict]) -> Any:
        try:
            formatted_context = "\n\n".join(
                [
                    f"Document {i+1}:\n{doc}"
                    for i, doc in enumerate(
                        element["text"] for element in context
                    )
                ]
            )
            m = f"""Answer the question based on the following context.

            <context>
            {formatted_context}
            </context>

            <question>
            {prompt}
            </question>"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": m}],
                options={"num_ctx": 4096, "temperature": 0.2},
            )
            print(f"Response: {response['message']['content']}")
            return {
                "retrieved_chunks": context,
                "answer": response["message"]["content"],
            }
        except Exception as e:
            raise Exception(f"Error in chat: {e}")

    def answer_dataset(
        self, student_search_result_path: str, save_directory: str
    ) -> None:
        try:
            dataset: StudentSearchResults = load_student_search_results(
                student_search_result_path
            )
            answered_results: list[MinimalAnswer] = []
            for item in dataset.search_results:
                formatted_context = "\n\n".join(
                    [
                        f"Document {i+1}:\n{doc}"
                        for i, doc in enumerate(
                            element.text for element in item.retrieved_sources
                        )
                    ]
                )

                m = f"""Answer the question based on the following context.

                <context>
                {formatted_context}
                </context>

                <question>
                {item.question}
                </question>"""

                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": m}],
                    options={"num_ctx": 4096, "temperature": 0.2},
                )
                answered_results.append(
                    MinimalAnswer(
                        question_id=item.question_id,
                        question=item.question,
                        retrieved_sources=item.retrieved_sources,
                        answer=response["message"]["content"],
                    )
                )
            return StudentSearchResultsAndAnswer(
                search_results=answered_results,
                k=dataset.k,
            )
        except Exception as e:
            raise Exception(f"Error in answer_dataset: {e}")
