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
            if len(chunk["chunk"]) <= 200:
                continue
            cleaned_chunks.append(chunk)
        return cleaned_chunks

    # TODO faire une fonction qui check si tout fait bien moins de 2000

    def format_chunks(self, chunks: list[str]) -> list[str]:
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

    def get_files_and_chunks(self):
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
                        # TODO faire une fonction pour recuperer le chemin mais que depuis le dossier du vllmpath
                        chunks.extend(self.chunk_text(content, file.resolve()))
                    elif file.suffix == ".md":
                        chunks.extend(self.chunk_md(content, file.resolve()))
                    elif file.suffix == ".py":
                        chunks.extend(self.chunk_code(content, file.resolve()))

        chunks = self.format_chunks(chunks)
        with open("chunks.json", "w") as f:
            json.dump(chunks, f, indent=4, default=str)

    def chunk_text(self, text, file_path):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(text)
        rslt = [{"file_path": file_path, "chunk": chunk} for chunk in chunks]
        return rslt

    def chunk_md(self, md, file_path):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["#", "##", "###", "####", "\n\n", "\n", "|", " ", ""],
        )
        chunks = splitter.split_text(md)
        rslt = [{"file_path": file_path, "chunk": chunk} for chunk in chunks]
        return rslt

    def chunk_code(self, code, file_path):
        tree = ast.parse(code)
        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if len(ast.dump(node)) > 2000:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=200,
                        separators=[
                            "\n\n",
                            "\n",
                            " ",
                            "",
                        ],
                    )
                    chunks.extend(splitter.split_text(ast.dump(node)))
                else:
                    chunks.append(ast.dump(node))
        rslt = [{"file_path": file_path, "chunk": chunk} for chunk in chunks]
        return rslt
