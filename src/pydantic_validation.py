from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, List, Optional
import uuid


class MinimalSource(BaseModel):
    """A single source location: a file and the character span it covers."""

    model_config = ConfigDict(extra="ignore")

    file_path: str
    first_character_index: int
    last_character_index: int
    text: Optional[str] = None


class UnansweredQuestion(BaseModel):
    """A question without its ground-truth sources/answer attached."""

    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and reference answer."""

    model_config = ConfigDict(extra="ignore")
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not."""

    model_config = ConfigDict(extra="ignore")
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

    @model_validator(mode="before")
    @classmethod
    def parse_questions(cls, data: Any) -> Any:
        """Parse each entry as `AnsweredQuestion` if it has sources/answer,
        or `UnansweredQuestion` otherwise, before the default union
        validation (which would otherwise silently coerce every entry to
        `UnansweredQuestion`, the first matching member of the union).
        """
        if isinstance(data, dict) and "rag_questions" in data:
            parsed = []
            for q in data["rag_questions"]:
                if "sources" in q and "answer" in q:
                    parsed.append(
                        AnsweredQuestion(
                            **{
                                k: v
                                for k, v in q.items()
                                if k in AnsweredQuestion.model_fields
                            }
                        )
                    )
                else:
                    parsed.append(
                        UnansweredQuestion(
                            **{
                                k: v
                                for k, v in q.items()
                                if k in UnansweredQuestion.model_fields
                            }
                        )
                    )
            data["rag_questions"] = parsed
        return data


class MinimalSearchResults(BaseModel):
    """The retrieved sources for a single question, without an answer."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources for a question, plus the generated answer."""

    model_config = ConfigDict(extra="ignore")

    answer: str


class StudentSearchResults(BaseModel):
    """Output of `search_dataset`: one `MinimalSearchResults` per question."""

    model_config = ConfigDict(extra="ignore")

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Output of `answer_dataset`: one `MinimalAnswer` per question."""

    model_config = ConfigDict(extra="ignore")

    search_results: List[MinimalAnswer]


def load_rag_dataset(dataset_path: str) -> RagDataset:
    """Load and validate a `RagDataset` from a JSON file."""
    with open(dataset_path, "r") as file:
        return RagDataset.model_validate_json(file.read())


def load_student_search_results(dataset_path: str) -> StudentSearchResults:
    """Load and validate a `StudentSearchResults` from a JSON file."""
    with open(dataset_path, "r") as file:
        return StudentSearchResults.model_validate_json(file.read())
