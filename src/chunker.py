import ast
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


class Chunker:
    def __init__(self, vllm_path: str):
        self.vllm_path = vllm_path

    def clean_hub(self, chunks: list[str]) -> list[str]:
        cleaned_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk.strip()) <= 200:
                continue
            cleaned_chunks.append(chunk)
        return cleaned_chunks

    def format_chunks(self, chunks: list[str]) -> list[str]:
        formated = [
            re.sub(
                r"\s*([,.;:!?()])\s*", r"\1", re.sub(r"\s+", " ", text).strip()
            )
            for text in chunks
        ]
        return self.clean_hub(formated)

    def get_files_and_chunks(self):
        ignored: list = ["setup.py"]
        authorized: list = [".txt", ".py", ".md", ".rst"]
        chunks: list[str] = []

        for file in Path(self.vllm_path).rglob("*"):
            print(f"Processing file: {file.name}")
            if file.name in ignored:
                continue
            if file.suffix in authorized:
                with open(file, "r", encoding="utf-8") as f:
                    content: str = f.read()
                    if content.strip() == "":
                        continue
                    if file.suffix == ".txt":
                        chunks.extend(self.chunk_text(content))
                    elif file.suffix == ".md":
                        chunks.extend(self.chunk_md(content))
                    # elif file.suffix == ".py":
                    #     chunks.extend(self.chunk_code(content))

        chunks = self.format_chunks(chunks)
        with open("chunks.json", "w") as f:
            json.dump(chunks, f, indent=4)

    def chunk_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def chunk_md(self, md):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["#", "##", "###", "####", "\n\n", "\n", "|", " ", ""],
        )
        return splitter.split_text(md)

    def chunk_code(self, code):
        pass
