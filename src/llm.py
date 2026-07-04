from __future__ import annotations

from typing import Any
import ollama

from pydantic_validation import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    load_student_search_results,
)


class Llm:
    def __init__(self) -> None:
        self.model = "qwen3:0.6b"

    def chat(self, prompt: str, context: list[dict]) -> Any:
        formatted_context = "\n\n".join(
            [
                f"Document {i+1}:\n{doc}"
                for i, doc in enumerate(element["text"] for element in context)
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

            output_dataset = StudentSearchResultsAndAnswer(
                search_results=answered_results,
                k=dataset.k,
            )
            output_path = f"{save_directory}/answered_dataset.json"
            with open(output_path, "w") as outfile:
                outfile.write(output_dataset.model_dump_json(indent=4))
            print(f"Answered dataset saved to {output_path}")
        except Exception as e:
            print(f"Error in answer_dataset: {e}")
