from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TypedDict
from pathlib import Path
from tqdm import tqdm
import json
import ast
import os
import re


class Chunk(TypedDict):
    """A single indexable slice of a source file."""
    text: str
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunker:
    """Walks a source tree and turns it into a flat list of `Chunk`."""

    def __init__(self, vllm_path: str) -> None:
        """Args:
        vllm_path: Root directory of the corpus to chunk.
        """
        self.vllm_path = vllm_path
        self.chunk_size = 2000

    def create_chunks(
        self,
        text: str,
        file_path: str,
        splitter: RecursiveCharacterTextSplitter,
        base_offset: int = 0,
    ) -> list[Chunk]:
        """Split `text` and resolve each piece's file-absolute offsets.

        Args:
            text: Text to split. May be the whole file, or an excerpt
                (e.g. one function's source) extracted from a larger file.
            file_path: Source file path recorded on each produced chunk.
            splitter: Splitter used to break `text` into pieces.
            base_offset: Absolute position of `text` within the real file.
                Added to every locally found index so offsets stay valid
                when `text` is only an excerpt rather than the full file.

        Returns:
            Chunks with `first_character_index`/`last_character_index`
            expressed relative to the original file.
        """
        chunks = splitter.split_text(text)

        rslt: list[Chunk] = []
        cursor: int = 0

        for chunk in chunks:
            first = text.find(chunk, cursor)

            if first == -1:
                first = text.find(chunk)

            last = first + len(chunk)
            cursor = last

            rslt.append(
                {
                    "first_character_index": base_offset + first,
                    "last_character_index": base_offset + last,
                    "file_path": file_path,
                    "text": chunk,
                }
            )

        return rslt

    @staticmethod
    def _line_offsets(text: str) -> list[int]:
        """Return the absolute character offset of the start of each line.

        Used to convert an AST node's `(lineno, col_offset)` into a single
        absolute character index within `text`.
        """
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def clean_hub(self, chunks: list[Chunk]) -> list[Chunk]:
        """Drop chunks shorter than 10% of `self.chunk_size`."""
        cleaned_chunks: list[Chunk] = []
        for chunk in chunks:
            if len(chunk["text"]) <= self.chunk_size * 0.10:
                continue
            cleaned_chunks.append(chunk)
        return cleaned_chunks

    def format_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Normalize whitespace/punctuation spacing, then drop tiny chunks."""
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
        """Chunk every eligible file under `self.vllm_path`.

        Dispatches each file to the chunker matching its extension
        (`.py` -> `chunk_code`, `.md` -> `chunk_md`, `.txt`/`.rst` ->
        `chunk_text`), then persists the result to
        `data/processed/chunk/chunks.json`.
        """
        ignored: list[str] = ["setup.py"]
        authorized: list[str] = [".txt", ".py", ".md", ".rst"]
        chunks: list[Chunk] = []

        for file in tqdm(
            list(Path(self.vllm_path).rglob("*")),
            desc="Scanning files",
        ):
            if file.name in ignored:
                continue
            if file.suffix in authorized:
                with open(file, "r") as f:
                    content: str = f.read()
                    if content.strip() == "":
                        continue
                    if file.suffix == ".txt" or file.suffix == ".rst":
                        chunks.extend(self.chunk_text(content, str(file)))
                    elif file.suffix == ".md":
                        chunks.extend(self.chunk_md(content, str(file)))
                    elif file.suffix == ".py":
                        chunks.extend(self.chunk_code(content, str(file)))

        chunks = self.format_chunks(chunks)
        file_path = "data/processed/chunk/chunks.json"
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open("./data/processed/chunk/chunks.json", "w") as f:
            json.dump(chunks, f, indent=4, default=str)

    def chunk_text(self, text: str, file_path: str) -> list[Chunk]:
        """Chunk plain text/reStructuredText with a recursive splitter."""
        splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=int(self.chunk_size * 0.75),
                chunk_overlap=int(self.chunk_size * 0.10),
                add_start_index=True,
                separators=["\n\n", "\n", " ", ""],
            )
        )
        return self.create_chunks(text, file_path, splitter)

    def chunk_md(self, md: str, file_path: str) -> list[Chunk]:
        """Chunk Markdown, preferring to split on heading boundaries."""
        splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=int(self.chunk_size * 0.75),
                chunk_overlap=int(self.chunk_size * 0.10),
                add_start_index=True,
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
        return self.create_chunks(md, file_path, splitter)

    def chunk_code(self, code: str, file_path: str) -> list[Chunk]:
        """Chunk Python source function-by-function and class-by-class.

        Each top-level or nested `def`/`class` is extracted via `ast` and
        chunked on its own, with offsets mapped back to their true
        position in `code` (via `_line_offsets`) so retrieved chunks stay
        traceable to an exact, verifiable span of the source file. Falls
        back to `chunk_text` if the file fails to parse, or if it has no
        function/class definitions at all.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self.chunk_text(code, file_path)

        rslt: list[Chunk] = []
        line_offsets = self._line_offsets(code)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(self.chunk_size * 0.75),
            chunk_overlap=int(self.chunk_size * 0.25),
            separators=["\n\n", "\n", " ", ""],
        )

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            segment = ast.get_source_segment(code, node)
            if not segment:
                continue
            node_start = line_offsets[node.lineno - 1] + node.col_offset
            rslt.extend(
                self.create_chunks(
                    segment, file_path, splitter, base_offset=node_start
                )
            )

        if not rslt:
            return self.chunk_text(code, file_path)

        return rslt
