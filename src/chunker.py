from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import json
import ast
import re


class Chunker:
    def __init__(self, vllm_path: str):
        self.vllm_path = vllm_path

    def clean_hub(self, chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned_chunks: list[dict[str, str]] = []
        for chunk in chunks:
            if len(chunk["chunk"]) <= 200:
                continue
            cleaned_chunks.append(chunk)
        return cleaned_chunks

    def check_chunk_length(self, chunks: list[dict[str, str]]) -> bool:
        for chunk in chunks:
            if len(chunk["chunk"]) > 2000:
                return False
        return True

    def format_chunks(
        self, chunks: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        formated = [
            {
                "file_path": chunk["file_path"],
                "chunk": re.sub(
                    r"\s*([,.;:!?()])\s*",
                    r"\1",
                    re.sub(r"\s+", " ", chunk["chunk"]).strip(),
                ),
            }
            for chunk in chunks
        ]
        return self.clean_hub(formated)

    def get_files_and_chunks(self) -> None:
        ignored: list[str] = ["setup.py"]
        authorized: list[str] = [".txt", ".py", ".md", ".rst"]
        chunks: list[dict[str, str]] = []

        for file in Path(self.vllm_path).rglob("*"):
            if file.name in ignored:
                continue
            if file.suffix in authorized:
                with open(file, "r", encoding="utf-8") as f:
                    content: str = f.read()
                    if content.strip() == "":
                        continue
                    if file.suffix == ".txt" or file.suffix == ".rst":
                        chunks.extend(
                            self.chunk_text(content, str(file.resolve()))
                        )
                    elif file.suffix == ".md":
                        chunks.extend(
                            self.chunk_md(content, str(file.resolve()))
                        )
                    elif file.suffix == ".py":
                        chunks.extend(
                            self.chunk_code(content, str(file.resolve()))
                        )

        chunks = self.format_chunks(chunks)
        # print(
        #     f"Everything is under 2000 char: {self.check_chunk_length(chunks)}"
        # )

        with open("chunks.json", "w") as f:
            json.dump(chunks, f, indent=4, default=str)

    def chunk_text(self, text: str, file_path: str) -> list[dict[str, str]]:
        splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""],
            )
        )
        chunks: list[str] = splitter.split_text(text)
        rslt: list[dict[str, str]] = [
            {"file_path": file_path, "chunk": chunk} for chunk in chunks
        ]
        return rslt

    def chunk_md(self, md: str, file_path: str) -> list[dict[str, str]]:
        splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=[
                    "#",
                    "##",
                    "###",
                    "####",
                    "\n\n",
                    "\n",
                    "|",
                    " ",
                    "",
                ],
            )
        )
        chunks: list[str] = splitter.split_text(md)
        rslt: list[dict[str, str]] = [
            {"file_path": file_path, "chunk": chunk} for chunk in chunks
        ]
        return rslt

    def chunk_code(self, code: str, file_path: str) -> list[dict[str, str]]:
        tree = ast.parse(code)
        chunks: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if len(ast.dump(node)) > 2000:
                    splitter: RecursiveCharacterTextSplitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=1500,
                            chunk_overlap=200,
                            separators=[
                                "\n\n",
                                "\n",
                                " ",
                                "",
                            ],
                        )
                    )
                    chunks.extend(splitter.split_text(ast.dump(node)))
                else:
                    chunks.append(ast.dump(node))
        rslt: list[dict[str, str]] = [
            {"file_path": file_path, "chunk": chunk} for chunk in chunks
        ]
        return rslt
