from typing import Any
import ollama


class Llm:
    def __init__(self) -> None:
        self.model = "qwen3:0.6b"

    def chat(self, prompt: str, context: list[str]) -> Any:
        formatted_context = "\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)]
        )
        message = f"""Answer the question based on the following context.

        <context>
        {formatted_context}
        </context>

        <question>
        {prompt}
        </question>"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            options={"num_ctx": 4096, "temperature": 0.2},
        )
        print(f"Response: {response['message']['content']}")
        return response["message"]["content"]
