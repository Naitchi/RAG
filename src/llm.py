from typing import Any
import ollama
import json


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

    def answer_dataset(
        self, student_search_result_path: str, save_directory: str
    ) -> None:
        try:
            with open(student_search_result_path, "r") as file:
                dataset = json.load(file)
            for item in dataset:
                prompt = item.get("prompt", "")
                retrieved_chunks = item.get("retrieved_chunks", [])
                answer = self.chat(prompt, retrieved_chunks)
                item["answer"] = answer
            output_path = f"{save_directory}/answered_dataset.json"
            with open(output_path, "w") as outfile:
                json.dump(dataset, outfile, indent=4)
            print(f"Answered dataset saved to {output_path}")
        except Exception as e:
            print(f"Error in answer_dataset: {e}")
