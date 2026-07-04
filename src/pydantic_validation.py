from pydantic import BaseModel, ConfigDict, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str  # TODO a voir y'a un grand monde ou ca me casse pas mal de trucs


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


def dump_model_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=4)
