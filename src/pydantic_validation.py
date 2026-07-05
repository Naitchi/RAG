from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, List, Optional
import uuid


class MinimalSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: str
    first_character_index: int
    last_character_index: int
    # TODO a voir y'a un grand monde ou ca me casse pas mal de trucs
    text: Optional[str] = None


class UnansweredQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    model_config = ConfigDict(extra="ignore")
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

    @model_validator(mode="before")
    @classmethod
    def parse_questions(cls, data: Any) -> Any:
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
    model_config = ConfigDict(extra="ignore")

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    model_config = ConfigDict(extra="ignore")

    answer: str


class StudentSearchResults(BaseModel):
    model_config = ConfigDict(extra="ignore")

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    model_config = ConfigDict(extra="ignore")

    search_results: List[MinimalAnswer]


def load_rag_dataset(dataset_path: str) -> RagDataset:
    with open(dataset_path, "r") as file:
        return RagDataset.model_validate_json(file.read())


def load_student_search_results(dataset_path: str) -> StudentSearchResults:
    with open(dataset_path, "r") as file:
        return StudentSearchResults.model_validate_json(file.read())
