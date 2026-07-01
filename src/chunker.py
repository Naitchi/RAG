from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict
from pathlib import Path
import json
import ast
import os
import re


class Chunk(TypedDict):
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunker:
    def __init__(self, vllm_path: str):
        self.vllm_path = vllm_path
        self.chunk_size = 2000

    def clean_hub(self, chunks: list[Chunk]) -> list[Chunk]:
        cleaned_chunks: list[Chunk] = []
        for chunk in chunks:
            if len(chunk["text"]) <= self.chunk_size * 0.10:
                continue
            cleaned_chunks.append(chunk)
        return cleaned_chunks

    def check_chunk_length(self, chunks: list[Chunk]) -> bool:
        for chunk in chunks:
            if len(chunk["text"]) > self.chunk_size:
                return False
        return True

    def format_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        formated: list[Chunk] = [
            {
                "first_character_index": chunk["first_character_index"],
                "last_character_index": chunk["last_character_index"],
                "file_path": chunk["file_path"],
                "text": re.sub(
                    r"\s*([,.;:!?()])\s*",
                    r"\1",
                    re.sub(r"\s+", " ", chunk["text"]).strip(),
                ),
            }
            for chunk in chunks
        ]
        return self.clean_hub(formated)

    def get_files_and_chunks(self) -> None:
        ignored: list[str] = ["setup.py"]
        authorized: list[str] = [".txt", ".py", ".md", ".rst"]
        chunks: list[Chunk] = []

        for file in Path(self.vllm_path).rglob("*"):
            if file.name in ignored:
                continue
            if file.suffix in authorized:
                with open(file, "r") as f:
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
        print(
            f"Everything is under {self.chunk_size}"
            f" char: {self.check_chunk_length(chunks)}"
        )
        file_path = "data/processed/chunk/chunks.json"
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open("./data/processed/chunk/chunks.json", "w") as f:
            json.dump(chunks, f, indent=4, default=str)

    def chunk_text(self, text: str, file_path: str) -> list[Chunk]:
        rslt: list[Chunk] = []
        start = 0
        splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 0.75,
                chunk_overlap=self.chunk_size * 0.10,
                separators=["\n\n", "\n", " ", ""],
            )
        )
        chunks: list[str] = splitter.split_text(text)
        for chunk in chunks:
            first = text.find(chunk, start)
            last = first + len(chunk)
            start = last
            rslt.append(
                {
                    "first_character_index": first,
                    "last_character_index": last,
                    "file_path": file_path,
                    "text": chunk,
                }
            )
        return rslt

    def chunk_md(self, md: str, file_path: str) -> list[Chunk]:
        splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size * 0.75,
                chunk_overlap=self.chunk_size * 0.10,
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
        rslt: list[Chunk] = []
        start = 0
        for chunk in chunks:
            first = md.find(chunk, start)
            last = first + len(chunk)
            start = last
            rslt.append(
                {
                    "first_character_index": first,
                    "last_character_index": last,
                    "file_path": file_path,
                    "text": chunk,
                }
            )
        return rslt

    def chunk_code(self, code: str, file_path: str) -> list[Chunk]:
        tree = ast.parse(code)
        chunks: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if len(ast.dump(node)) > self.chunk_size:
                    splitter: RecursiveCharacterTextSplitter = (
                        RecursiveCharacterTextSplitter(
                            chunk_size=self.chunk_size * 0.75,
                            chunk_overlap=self.chunk_size * 0.10,
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
        rslt: list[Chunk] = []
        start = 0
        for chunk in chunks:
            first = code.find(chunk, start)
            last = first + len(chunk)
            start = last
            rslt.append(
                {
                    "first_character_index": first,
                    "last_character_index": last,
                    "file_path": file_path,
                    "text": chunk,
                }
            )
        return rslt
