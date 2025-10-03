from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    query: str


class Snippet(BaseModel):
    text: str
    score: float


class AnswerResponse(BaseModel):
    snippets: List[Snippet]
    answer: str
